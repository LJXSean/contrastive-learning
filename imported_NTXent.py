#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import json
import torch
from torch.optim import AdamW
from transformers import AutoModel
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from pytorch_metric_learning import miners, losses
from datasets import load_metric
from torch import nn
import os


# #### Tokenise data

# In[3]:


class CitationDataSet:
    def __init__(self, source, tokenizer_name='allenai/scibert_scivocab_uncased'):
        self.dataset = load_dataset("csv", data_files=source)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.column_names = self.dataset['train'].column_names
        self.col_name = self.column_names[0]
        self.label_col = self.column_names[-1]

    def tokenize(self, examples, max_length=256):
        id_masks_all_cols = []
        batch_size = len(examples[self.col_name])

        # Tokenize examples for each column (ignore last column ie. 'label')
        for k in self.column_names[:-1]:
            id_mask = self.tokenizer(examples[k], truncation=True, padding='max_length', max_length=max_length)
            id_masks_all_cols.append(id_mask)

        zipped_id_mask = {}
        id_mask_col = id_masks_all_cols[0]

        # Zips all columns together for each feature, input_id/attention_mask
        for feature in id_mask_col:
            zipped_id_mask[feature] = [[id_mask[feature][i] for id_mask in id_masks_all_cols] for i in range(batch_size)]

        zipped_id_mask[self.label_col] = examples[self.label_col]
        return zipped_id_mask

    
    def get_dataloader(self):
        # Shape = [features, batch_size, (anchor, pos)/label]
        dataset = self.dataset['train'].map(self.tokenize, batched=True, remove_columns=self.column_names)

        dataset.set_format("torch")
        train_dataloader = DataLoader(dataset, shuffle=True, batch_size=5)
        return train_dataloader


# #### Fine Tune Model

# In[4]:


# Uses [CLS] token representation
class CitationIntentEncoder(nn.Module):
    def __init__(self, sciBert, dropout_p=0.5):
        super(CitationIntentEncoder, self).__init__()
        self.sentence_transformer = sciBert
        self.dropout = nn.Dropout(dropout_p)
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask):
        embeddings = self.sentence_transformer(input_ids, attention_mask)
        cls_representation = embeddings.last_hidden_state[:, 0]
        cls_representation = self.dropout(cls_representation)
        x = self.dense(cls_representation)
        return self.activation(x)

def encoder(batch, model):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label']

    # Flatten to encode all at once
    input_ids = torch.cat((input_ids[:, 0], input_ids[:, 1]))
    attention_mask = torch.cat((attention_mask[:, 0], attention_mask[:, 1]))
    labels = labels.repeat(2)

    # Data augmentation handled by scibert, dropout implemented under the hood
    embeddings = model(input_ids, attention_mask)
    return embeddings, labels


# In[5]:


miner = miners.MultiSimilarityMiner()
loss_func = losses.NTXentLoss(temperature=0.07)

def train_and_save(save_directory, train_dataloader, mining=False, model_name='allenai/scibert_scivocab_uncased'):
    sciBert = AutoModel.from_pretrained(model_name)
    model = CitationIntentEncoder(sciBert)
    
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 3

    for epoch in range(epochs):
        total_loss = 0
        # Shape = [#features, #batch_size, #tensor_length]
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            embeddings, labels = encoder(batch, model)
            if mining:
                hard_pairs = miner(embeddings, labels)
                loss = loss_func(embeddings, labels, hard_pairs)
            else:
                loss = loss_func(embeddings, labels)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Batch: {i+1}/{len(train_dataloader)}")
            break
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}")
        break

    # Save the configuration of SciBERT separately
    torch.save(model.state_dict(), save_directory + '/CLModel_state_dict.bin')
    model.sentence_transformer.config.save_pretrained(save_directory)
    return model


# In[8]:


save_directory = './sectionPaper_mlp_without_hard'
if not os.path.isdir(save_directory):
    os.mkdir(save_directory)

train_dataloader = CitationDataSet("data_file_sectionPaper.csv").get_dataloader()
trained_model = train_and_save(save_directory, train_dataloader, True)


# #### Sanity Check

# In[ ]:


# Load trained model
config = AutoConfig.from_pretrained('./sectionPaper_mlp_without_hard')
sciBert = AutoModel.from_config(config)
new_model = CitationIntentEncoder(sciBert)

new_model.load_state_dict(torch.load('sectionPaper_mlp_without_hard/CLModel_state_dict.bin'))


# In[ ]:


sample_batch = None
for i, batch in enumerate(train_dataloader):
    sample_batch = batch
    break

trained_model.eval()
with torch.no_grad():
    embeddings, labels = encoder(sample_batch, trained_model)
    print(embeddings[0][:10])
    print(labels[0])


# In[ ]:


new_model.eval()
with torch.no_grad():
    embeddings, labels = encoder(sample_batch, new_model)
    print(embeddings[0][:10])
    print(labels[0])

