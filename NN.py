import pandas as pd
import json

data_path_aclarc = "./acl-arc/scaffolds/sections-scaffold-train.jsonl"
data_path_scicite = "./scicite/scaffolds/sections-scaffold-train.jsonl"
with open(data_path_aclarc, encoding='utf-8') as data_file:
    data = [json.loads(line) for line in data_file]
    df = pd.DataFrame(data).drop_duplicates()

sort_cols_section_paper = ['section_name', 'cited_paper_id']
sort_cols_section = ['section_name']
sort_cols = sort_cols_section_paper
final_cols = ['text', 'text_pos', 'section_name', 'citing_paper_id', 'cited_paper_id']

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
def get_pos_samples_concat(df):
    df_concat = df.copy(deep=True)

    # Dummy columns for groupby, to keep original columns
    include_groups = [i + '_drop' for i in sort_cols]
    df_concat[include_groups] = df_concat[sort_cols]
    
    result = df_concat.groupby(include_groups).apply(split_and_concatenate, include_groups=False).reset_index(drop=True)
    return result

concat = get_pos_samples_concat(df)

# Replace NA with text_pos (dropout in roberta will treat this as unsupervised learning)
def handle_na(df):
    df.loc[pd.isna(df['text']), 'text'] = df.loc[pd.isna(df['text'])]['text_pos']

handle_na(concat)

concat[['text', 'text_pos']].to_csv('data_file.csv', index=False)

from transformers import AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("csv", data_files="data_file.csv")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
column_names = dataset['train'].column_names

# Testing
def tokenize(examples, max_length=256):
    id_masks_all_cols = []
    col_name = column_names[0]
    length = len(examples[col_name])

    # Tokenize examples for each column
    for k in column_names:
        id_mask = tokenizer(examples[k], truncation=True, padding='max_length', max_length=max_length)
        id_masks_all_cols.append(id_mask)

    zipped_id_mask = {}
    id_mask_col = id_masks_all_cols[0]

    # Zips all columns together for each feature, input_id/attention_mask
    for feature in id_mask_col:
        zipped_id_mask[feature] = [[id_mask[feature][i] for id_mask in id_masks_all_cols] for i in range(length)]

    return zipped_id_mask

# Shape = [#features, #sentences, #samples(anchor, pos, neg)]
tokenized = dataset['train'].map(tokenize, batched=True, remove_columns=column_names)

from torch.utils.data import DataLoader

tokenized.set_format("torch")

small_train_dataset = tokenized.shuffle(seed=42).select(range(6144))
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=32)
#train_dataloader = DataLoader(tokenized, shuffle=True, batch_size=32)

for batch in train_dataloader:
    # Shape = [#featuress, #batch_size, #tensor_length]
    print(batch['input_ids'].shape)
    break

import torch
import torch.nn as nn

def contrastive_loss(embeddings, temperature=0.1):
    sents_per_vector = embeddings.size(1)

    if sents_per_vector < 2 or sents_per_vector > 3:
        raise Exception("Unexpected number of sentences per sample received. Expected: 2/3") 
    
    cos_sim = nn.CosineSimilarity(dim=-1)

    # Reshape to 3D for broadcast computation
    anchor = embeddings[:, 0].unsqueeze(1)
    positive = embeddings[:, 1].unsqueeze(0)
    
    # Pairwise cosine similarity, shape = [batch_size, batch_size]
    pairwise_sim = cos_sim(anchor, positive)

    # index of positive sample for corresponding anchors (matrix diagonal)
    target = torch.arange(pairwise_sim.size(0))

    # Horizontally concatenate hard_neg similarities (if any)
    if sents_per_vector == 3:
        hard_neg = embeddings[:, 2].unsqueeze(0)
        hard_neg_sim = cos_sim(anchor, hard_neg)
        pairwise_sim = torch.cat([pairwise_sim, hard_neg_sim], 1)
    
    pairwise_sim /= temperature

    loss = nn.CrossEntropyLoss()
    output = loss(pairwise_sim, target)

    return output

from torch.optim import AdamW
from transformers import RobertaModel
import time

def train(batch):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    batch_size, sents_per_vector, tensor_size = input_ids.shape

    # Flatten to encode all at once
    input_ids = torch.reshape(input_ids, (-1, tensor_size))
    attention_mask = torch.reshape(attention_mask, (-1, tensor_size))

    # Use [CLS] token representation
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state[:, 0]

    # Reshape back to nested tensors
    embeddings = torch.reshape(embeddings, (batch_size, sents_per_vector, -1))
    return embeddings

model = RobertaModel.from_pretrained('roberta-base')
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 2
start = time.time()
for epoch in range(epochs):
    total_loss = 0
    # Shape = [#features, #batch_size, #tensor_length]
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        embeddings = train(batch)
        loss = contrastive_loss(embeddings)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
  
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}")

save_directory = './large-pretrained'  # Specify your save directory
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("time taken = {time.time()-start}")
