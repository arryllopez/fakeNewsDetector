import pandas as pd
import torch
import numpy as np
import nltk 
import sklearn
import transformers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm  


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

class LIARDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs['input_ids'][idx],
            'attention_mask': self.inputs['attention_mask'][idx],
            'labels': self.labels[idx]
        }




# Create datasets
train_dataset = LIARDataset(X_train, y_train)
valid_dataset = LIARDataset(X_valid, y_valid)
test_dataset = LIARDataset(X_test, y_test)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)




# Load pre-trained model with a classification head
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',  # Replace with your chosen transformer model
    num_labels=6          # Number of classes in the LIAR dataset
)




optimizer = AdamW(model.parameters(), lr=5e-5)  # Adjust learning rate as needed
loss_fn = torch.nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)




# Training loop
epochs = 3  # Number of epochs
for epoch in range(epochs):
    model.train()  # Set model to training mode
    total_loss = 0

    for batch in tqdm(train_loader):
        # Move data to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")


#EVALUTATION 
from sklearn.metrics import accuracy_score

model.eval()  
all_preds = []
all_labels = []

with torch.no_grad(): 
    for batch in valid_loader:
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, axis=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute accuracy (how well the model does on the valid dataset)
accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy:.4f}")

model.save_pretrained('./fake_news_model')
tokenizer.save_pretrained('./fake_news_model')
