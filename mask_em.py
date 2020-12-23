import json
import os
import argparse
import itertools

from statistics import mode, mean, stdev
from tqdm import tqdm
from nltk.tokenize import word_tokenize

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import *

parser = argparse.ArgumentParser(description='Entropy test of Masked language models')
parser.add_argument('--data', type=str, required=True,
                    help='location of the data file (text file with documents separated with two newlines \\n\\n )')
parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='Identifier of the model from huggingface')
parser.add_argument('--output', type=str, default='entropy_set.csv',
                    help='Output file name')
parser.add_argument('--batch_size', type=int, default=48,
                    help='Batch size per GPU')

def construct_chunks(args, lower=True):
    documents = open(args.data).read().split("\n\n")
    print("Processing", len(documents), "documents")

    samples = []
    for doc in tqdm(documents, desc="Preparing documents"):
        words = word_tokenize(doc.lower())
        for i, w in enumerate(words):
            
            if w in STOPW:
                continue

            subwords = tokenizer.tokenize(w)
            gold_sample = " ".join(words[max(i - CONTEXT_LEN // 2, 0): i + CONTEXT_LEN // 2])
            # mask all occurrences # masked_sample = gold_sample.replace(w, tokenizer.mask_token if len(subwords) == 1 else " ".join(len(subwords) * [tokenizer.mask_token, ] ) )
            masked_sample = " ".join( words[max(i - CONTEXT_LEN // 2, 0): i] + (len(subwords) * [tokenizer.mask_token, ]) + words[i + 1: i + CONTEXT_LEN // 2 ] )
            
            if len(tokenizer.tokenize(masked_sample)) > CONTEXT_LEN + 64:
                continue
            
            samples.append({ "word" : w, "subwords": subwords, "gold": gold_sample, "masked": masked_sample })

    print("Resulted", len(samples), "masked samples")

    return samples


def prepare_batches(samples, args):

    masked_samples = [ s["masked"] for s in samples ]
    gold_samples = [ s["gold"] for s in samples]

    inputs = tokenizer.batch_encode_plus(masked_samples,
                        padding='longest',
                        add_special_tokens=True,
                        pad_to_multiple_of=8,
                        return_tensors='pt')

    labels = tokenizer.batch_encode_plus(gold_samples,
                        padding='longest',
                        add_special_tokens=True,
                        pad_to_multiple_of=8,
                        return_tensors='pt')["input_ids"]

    masked_words_ids = [ [ tokenizer.vocab[s] for s in sample["subwords"] ] for sample in samples ]

    return inputs, labels, masked_words_ids


def predict(model, inputs, labels, words, subwords, masked_words_ids, batch_size, output_dict):
    
    data = TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], labels)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)


    model.eval()
    with torch.no_grad():
        i = 0
        for batch in dataloader:
            b_input_ids, b_input_mask, b_token_type_ids, b_labels = tuple(t.to(device) for t in batch)
            
            with torch.cuda.amp.autocast():
                output = model(b_input_ids, 
                attention_mask=b_input_mask,
                token_type_ids=b_token_type_ids)
                probs = torch.nn.functional.softmax(output[0].detach(), dim=2)
                entropies = -torch.sum(torch.log2(probs) * probs, dim=2)

            entropies, probs = entropies.cpu(), probs.cpu()
            b_mask_idx, t_mask_idx = torch.where((b_input_ids != b_labels).cpu())

            for j in range(b_input_ids.shape[0]):
                # p_word = probs[j, t_mask_idx[b_mask_idx == j]][:, masked_words_ids[i]]
                # output_dict[word]["replacements"] += tokenizer.decode(probs[j, t_mask_idx[b_mask_idx == j]].argmax(dim=1).tolist()).split()

                subword_set = subwords[i]
                for sub, sub_id in zip(subword_set, masked_words_ids[i]):
                    output_dict[sub]["entropies"] += entropies[j, t_mask_idx[b_mask_idx == j]].flatten().tolist()
                
                i += 1

    return output_dict 

CONTEXT_LEN = 256
use_gpu = True
device_ids = [0, ]

if __name__ == '__main__':
    args = parser.parse_args()

    # from collections import namedtuple
    # Argg = namedtuple('Argg', 'model output data')
    # args = Argg("bert-base-uncased", "entropy_set.json", "imdb.txt")

    print("Masking tool")
    print("Output will be written into", args.output)

    STOPW = set(open("stopwords.txt").read().splitlines())
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    masked_samples = construct_chunks(args)
    
    words = [ x["word"] for x in masked_samples ]
    subwords = [ x["subwords"] for x in masked_samples ]
    output = { w : { "entropies" : [] } for w in list(itertools.chain(*subwords)) }

    ## Initialize model
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:%d"%(device_ids[0]))
    else:
        device = torch.device("cpu")

    if len(device_ids) > 1 and device.type == "cuda":
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    for i in tqdm(range((len(masked_samples) // 1024) + 1), desc="Predicting"):
        ## tokenize
        inputs, labels, masked_words_ids = prepare_batches(masked_samples[i*1024:(i+1)*1024], args)

        ## predict
        batch_size = args.batch_size * len(device_ids)
        predict(model, inputs, labels, words, subwords, masked_words_ids, batch_size, output)

    ## sort
    for k in list(output.keys()):
        if len(output[k]["entropies"]) == 0:
            del output[k]
    
    output = list(map(lambda x: (x[0], mean(x[1]["entropies"])), output.items() ) ) 
    sorted_output = sorted(output, key=lambda x: x[1], reverse=True)

    ## write output
    with open(args.output, "w") as fo:
        import csv
        csvwriter = csv.writer(fo)
        csvwriter.writerows(sorted_output)
