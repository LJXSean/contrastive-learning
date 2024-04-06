#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset
from torch.optim import AdamW
from transformers import RobertaModel, RobertaTokenizer

from sklearn.metrics import f1_score


# In[2]:


file_path_train = 'scicite/train.jsonl'
file_path_dev = 'scicite/dev.jsonl'
file_path_test = 'scicite/test.jsonl'
train_data = []
dev_data = []
test_data = []
with open(file_path_train, 'r', encoding='utf-8') as file:
    for line in file:
        train_data.append(json.loads(line))
with open(file_path_dev, 'r', encoding='utf-8') as file:
    for line in file:
        dev_data.append(json.loads(line))
with open(file_path_test, 'r', encoding='utf-8') as file:
    for line in file:
        test_data.append(json.loads(line))

# In[3]:


class CitationsDatasetWithoutInputExample():
    label_to_id = {'background': 0, 'method': 1, 'result': 2}
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]['string'], CitationsDatasetWithoutInputExample.label_to_id[self.data[item]['label']]


# In[4]:


train_dataset = CitationsDatasetWithoutInputExample(train_data)
train_batch_size = 32
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)


# In[5]:


dev_dataset = CitationsDatasetWithoutInputExample(dev_data)
dev_batch_size = 32
dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=dev_batch_size)


# In[6]:


class CitationIntentClassifier(nn.Module):
    def __init__(self, model_path, num_labels):
        super(CitationIntentClassifier, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.sentence_transformer = RobertaModel.from_pretrained(model_path)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_texts):
        tokenised = self.tokenizer(input_texts, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
        embeddings = self.sentence_transformer(**tokenised)
        cls_representation = embeddings.last_hidden_state[:, 0]
        return self.classifier(cls_representation)
    


def train_epoch(model, dataloader, loss_func, optimizer):
    model.train()
    total_loss = 0
    for input_texts, labels in dataloader:
        optimizer.zero_grad()
        output = model(input_texts)
        loss = loss_func(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        break
        
    print(f"Training loss: {total_loss / len(dataloader)}")

def evaluate(model, dataloader, loss_func):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for input_texts, labels in dataloader:
            output = model(input_texts)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            total_correct += (output.argmax(1) == labels).sum().item()
            break
            
    print(f"Evaluation loss: {total_loss / len(dataloader)}")
    print(f"Evaluation accuracy: {total_correct / len(dataloader.dataset)}")

test_dataset = CitationsDatasetWithoutInputExample(test_data)
test_batch_size = 32
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=test_batch_size)

# In[8]:

def test(model, dataloader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for input_texts, labels in dataloader:
            output = model(input_texts)
            _, predicted_labels = torch.max(output, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            break
    return predictions, true_labels

def train_test_loop(model_name):
    num_labels = 3
    citation_intent_classifier = CitationIntentClassifier(model_name, num_labels)

    # Parameters
    learning_rate = 2e-5
    num_epochs = 5

    optimizer = torch.optim.Adam(citation_intent_classifier.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()


    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_epoch(citation_intent_classifier, train_dataloader, loss_func, optimizer)
        evaluate(citation_intent_classifier, dev_dataloader, loss_func)
        break

    predictions, true_labels = test(citation_intent_classifier, test_dataloader)

    f1 = f1_score(true_labels, predictions, average='macro')
    print(f"F1 Score for {model_name}: {f1}")

models = ['roberta-base', 'largest']
for model in models:
    train_test_loop(model)
