import pandas as pd
import torch
import numpy as np
import nltk 
import sklearn
import transformers

print("The libraries are working")


# testing hugging face model loading
from transformers import AutoModel, AutoTokenizer

# usign BERT as an example
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

print("Model and tokenizer loaded successfully!")




# teting if pytorch is workign alongside hugging face transformers
from transformers import AutoModel, AutoTokenizer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Test tokenization and model inference
sample_text = "Fake news detection is challenging."
inputs = tokenizer(sample_text, return_tensors="pt")
outputs = model(**inputs)

print("Model and tokenizer are working!")


#LOADING THE FIRST DATASET, LIAR DATASET FROM KAGGLE AND PROCESSING DATA
columns = ['id','label','text','subject','speaker','job title','state info','party','barely true','false','half true','mostly true','pants on fire','context']
label_map = {'pants-fire':-3, 'false':-2, 'barely-true':-1, 'half-true':1, 'mostly-true':2, 'true':3}

train = pd.read_csv('datasets/archive/train.tsv',sep='\t',header=None, names=columns)
test =  pd.read_csv('datasets/archive/test.tsv',sep='\t',header=None, names=columns)
valid = pd.read_csv('datasets/archive/valid.tsv',sep='\t',header=None, names=columns)


print(train.head())