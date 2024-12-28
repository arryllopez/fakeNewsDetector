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
