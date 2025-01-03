import pandas as pd
import torch
import numpy as np
import nltk 
import sklearn
import transformers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader


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

#BEGINNING TO TRAIN THE MODEL

# Map labels to numeric values (custom mapping if not already done)
label_map = {
    'pants-fire': 0, 'false': 1, 'barely-true': 2, 
    'half-true': 3, 'mostly-true': 4, 'true': 5
}
train['label'] = train['label'].map(label_map)
valid['label'] = valid['label'].map(label_map)
test['label'] = test['label'].map(label_map)

# Initialize tokenizer (you can replace 'bert-base-uncased' with another transformer model)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize_function(texts, max_length=512):
    return tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

# Tokenize the train, validation, and test data
X_train = tokenize_function(train['text'].tolist())
X_valid = tokenize_function(valid['text'].tolist())
X_test = tokenize_function(test['text'].tolist())

# Convert labels to tensors
y_train = torch.tensor(train['label'].values)
y_valid = torch.tensor(valid['label'].values)
y_test = torch.tensor(test['label'].values)