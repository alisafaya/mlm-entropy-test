# Masked LM Entropy testing tool

This tool extracts a list of words with their corresponding average entropies, using a masked language model like BERT. The idea was to mask each word individually, and to see how much the model is struggling to predict it, by calculating the entropy of that prediction.

## Usage:

```
python mask_em.py [-h] --data DATA [--model MODEL] [--output OUTPUT] [--batch_size BATCH_SIZE]

Entropy test of Masked language models

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data file (text file with documents separated with two newlines \n\n )
  --model MODEL         Identifier of the model from huggingface
  --output OUTPUT       Output file name
  --batch_size BATCH_SIZE  Batch size
```

**Example**: 


```shell
python mask_em.py --data imdb.txt
```

**Output sample**:

|word    |average-entropy   |
|--------|------------------|
|clothing|3.515076          |
|brother |3.506209          |
|complete|3.502324          |
|pool    |3.496363          |
|twin    |3.493093          |
|dogs    |3.488596          |
|party   |3.483214          |

