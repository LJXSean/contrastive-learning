#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import torch
from torch.optim import AdamW
from transformers import AutoModel
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from pytorch_metric_learning import miners, losses
from datasets import load_metric


# In[2]:


data_path_aclarc = "./acl-arc/scaffolds/sections-scaffold-train.jsonl"
data_path_scicite = "./scicite/scaffolds/sections-scaffold-train.jsonl"
with open(data_path_aclarc, encoding='utf-8') as data_file:
    data = [json.loads(line) for line in data_file]
    df = pd.DataFrame(data).drop_duplicates()


# #### Positive Sampling

# In[3]:


final_cols = ['text', 'text_pos', 'label']

def split_and_concatenate(group):
    # Calculate the split index
    split_index = len(group) // 2
    
    # Split the group into two halves
    first_half = group.iloc[:split_index].reset_index(drop=True)['text']
    second_half = group.iloc[split_index:].reset_index(drop=True)
    second_half.rename(columns={'text': 'text_pos'}, inplace=True)

    # Concatenate the halves horizontally
    concatenated = pd.concat([first_half, second_half], axis=1)
    return concatenated

# Gets samples using concatenation
def get_pos_samples_concat(df, sort_cols):
    df_concat = df.copy(deep=True)

    # Dummy columns for groupby, to keep original columns
    include_groups = [i + '_drop' for i in sort_cols]
    df_concat[include_groups] = df_concat[sort_cols]
    
    result = df_concat.groupby(include_groups).apply(split_and_concatenate, include_groups=False).reset_index(drop=True)
    return result

def add_label(result, sort_cols):
    # Add Label
    if len(sort_cols) > 1:
        result['combined'] = result[sort_cols].T.agg(''.join)
    else:
        result['combined'] = result[sort_cols]

    labels, _ = pd.factorize(result['combined'])
    result['label'] = labels

    return result[final_cols]

# Replace NA with text_pos (dropout in roberta will treat this as unsupervised learning)
def handle_na(df):
    df.loc[pd.isna(df['text']), 'text'] = df.loc[pd.isna(df['text'])]['text_pos']

def process_data(df, sort_cols):
    concat = get_pos_samples_concat(df, sort_cols=sort_cols)
    concat_with_labels = add_label(concat, sort_cols)
    handle_na(concat_with_labels)
    return concat_with_labels

section_paper = ['section_name', 'cited_paper_id']
section = ['section_name']

concat_section_paper = process_data(df, sort_cols=section_paper)
concat_section = process_data(df, sort_cols=section)


# #### Exploration

# In[4]:


print(concat_section_paper['label'].value_counts())
print(concat_section['label'].value_counts())


# In[5]:


concat_section_paper.to_csv('data_file_sectionPaper.csv', index=False)
concat_section.to_csv('data_file_section.csv', index=False)


# #### Tokenise data

# In[6]:


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
        train_dataloader = DataLoader(dataset, shuffle=True, batch_size=32)
        return train_dataloader


# #### Fine Tune Model

# In[7]:


# Uses [CLS] token representation
def encoder(batch, model):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label']

    # Flatten to encode all at once
    input_ids = torch.cat((input_ids[:, 0], input_ids[:, 1]))
    attention_mask = torch.cat((attention_mask[:, 0], attention_mask[:, 1]))
    labels = labels.repeat(2)

    # Data augmentation handled by scibert, dropout implemented under the hood
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state[:, 0]
    return embeddings, labels


# In[8]:


miner = miners.MultiSimilarityMiner()
loss_func = losses.NTXentLoss(temperature=0.07)

def train_and_save(save_directory, train_dataloader, mining=False, model_name='allenai/scibert_scivocab_uncased'):
    model = AutoModel.from_pretrained(model_name)
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 2

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

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}")
    
    model.save_pretrained(save_directory)


# In[9]:


train_dataloader = CitationDataSet("data_file_sectionPaper.csv").get_dataloader()
train_and_save('./sectionPaper_with_hard', train_dataloader, True)

"""
train_and_save('./sectionPaper_without_hard', train_dataloader, False)

train_dataloader = CitationDataSet('data_file_section.csv').get_dataloader()
train_and_save('./section_with_hard', train_dataloader, True)
train_and_save('./section_without_hard', train_dataloader, False)
"""


# In[ ]:




