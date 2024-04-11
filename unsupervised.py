#!/usr/bin/env python
# coding: utf-8


# In[3]:


import pandas as pd
import numpy as np
import json
import torch
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_metric_learning import miners, losses
from datasets import load_metric
import time
import torch
import math


# In[2]:


if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Device: mps")
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Device: cuda")
else:
    device = torch.device('cpu')
    print("Device: cpu")


# #### Tokenise data

# In[26]:


class CitationDataSet:
    def __init__(self, source, tokenizer_name='allenai/scibert_scivocab_uncased'):
        self.dataset = load_dataset("csv", data_files=source)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.column_names = self.dataset['train'].column_names

    def tokenize(self, examples, max_length=256):
        id_mask = self.tokenizer(examples['cleaned_cite_text'], truncation=True, padding='max_length', max_length=max_length)
        return id_mask

    def get_dataloader(self, batch_size=8):
        # Tokenize and set format for the dataset
        dataset = self.dataset['train'].map(self.tokenize, batched=True, remove_columns=self.column_names)
        dataset.set_format(type='torch')
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# #### Fine Tune Model

# In[20]:


# Uses [CLS] token representation
def encoder(batch, model):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    batch_size = input_ids.size(0)

    # Feed data into model twice (augmented using dropout)
    input_ids = input_ids.repeat(2, 1).to(device)
    attention_mask = attention_mask.repeat(2, 1).to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state[:, 0]

    original, augmented = embeddings[:batch_size], embeddings[batch_size:]
    return original, augmented


# In[27]:


NTXent = losses.NTXentLoss(temperature=0.07)
loss_func = losses.SelfSupervisedLoss(NTXent, symmetric=False)

def train_and_save(save_directory, train_dataloader, warmup=True, epochs=4, model_name='allenai/scibert_scivocab_uncased'):
    model = AutoModel.from_pretrained(model_name).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(train_dataloader) * epochs
    num_warmup_steps = math.ceil(num_training_steps * 0.1)
    if warmup:
      scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    for epoch in range(epochs):
        total_loss = 0
        # Shape = [#features, #batch_size, #tensor_length]
        start = time.time()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            embeddings, ref_emb = encoder(batch, model)
            loss = loss_func(embeddings, ref_emb)

            loss.backward()
            optimizer.step()
            if warmup:
              scheduler.step()

            total_loss += loss.item()

            if i % 1000 == 0:
                print(f"Batch: {i+1}/{len(train_dataloader)}")
            break

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}, Time Taken: {(time.time()-start)/60}")

    model.save_pretrained(save_directory)


# In[28]:


train_dataloader = CitationDataSet('./unsupervised.csv').get_dataloader(batch_size=32)
train_and_save('./unsupervised_sciciteSection_3epoch_32batch', train_dataloader, epochs=3, warmup=False)


# In[ ]:


import math
import random
import numpy as np
import json
import torch
from torch import nn
from collections import defaultdict
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.optim import AdamW

from sklearn.metrics import f1_score


# In[ ]:


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


# In[ ]:


class CitationsDatasetWithoutInputExample():
    label_to_id = {'background': 0, 'method': 1, 'result': 2}
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]['string'], CitationsDatasetWithoutInputExample.label_to_id[self.data[item]['label']]


# In[ ]:


batch_size = 16
train_dataset = CitationsDatasetWithoutInputExample(train_data)
train_batch_size = batch_size
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)


# In[ ]:


dev_dataset = CitationsDatasetWithoutInputExample(dev_data)
dev_batch_size = batch_size
dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=dev_batch_size)


# In[ ]:


class CitationIntentClassifier(nn.Module):
    def __init__(self, model_path, num_labels):
        super(CitationIntentClassifier, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.sentence_transformer = AutoModel.from_pretrained(model_path)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_texts):
        tokenised = self.tokenizer(input_texts, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
        tokenised = tokenised.to(device)
        embeddings = self.sentence_transformer(**tokenised)
        cls_representation = embeddings.last_hidden_state[:, 0]
        return self.classifier(cls_representation)

def train_epoch(model, dataloader, loss_func, optimizer):
    model.train()
    total_loss = 0
    for input_texts, labels in dataloader:
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(input_texts)
        loss = loss_func(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Training loss: {total_loss / len(dataloader)}")

def evaluate(model, dataloader, loss_func):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for input_texts, labels in dataloader:
            labels = labels.to(device)
            output = model(input_texts)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            total_correct += (output.argmax(1) == labels).sum().item()

    print(f"Evaluation loss: {total_loss / len(dataloader)}")
    print(f"Evaluation accuracy: {total_correct / len(dataloader.dataset)}")


# In[ ]:


test_dataset = CitationsDatasetWithoutInputExample(test_data)
test_batch_size = batch_size
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=test_batch_size)


# In[ ]:


def test(model, dataloader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for input_texts, labels in dataloader:
            labels = labels.to(device)
            output = model(input_texts)
            _, predicted_labels = torch.max(output, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return predictions, true_labels


def train_test_loop(model_name, num_epochs=5, learning_rate=2e-5):
    num_labels = 3
    citation_intent_classifier = CitationIntentClassifier(model_name, num_labels).to(device)

    optimizer = torch.optim.Adam(citation_intent_classifier.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_epoch(citation_intent_classifier, train_dataloader, loss_func, optimizer)
        evaluate(citation_intent_classifier, dev_dataloader, loss_func)

    predictions, true_labels = test(citation_intent_classifier, test_dataloader)
    f1 = f1_score(true_labels, predictions, average='macro')
    print(f"F1 Score for {model_name}: {f1}")


# In[ ]:


train_test_loop('./unsupervised_sciciteSection_3epoch_32batch')

