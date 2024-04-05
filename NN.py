from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import json

dataset = load_dataset("csv", data_files="data_file.csv")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
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

# 80-20 Train-Test split
train_size = int(0.8 * len(tokenized))
test_size = len(tokenized) - train_size

train_dataset = tokenized.shuffle(seed=42).select(range(train_size))
test_dataset = tokenized.shuffle(seed=42).select(range(train_size, train_size+test_size))

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=64)

import torch
import torch.nn as nn

def contrastive_loss(embeddings, temperature=0.1, train=True):
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

    if train:
        loss = nn.CrossEntropyLoss()
        output = loss(pairwise_sim, target)

        return output
    else:
        predicted = torch.argmax(pairwise_sim, dim=1)
        return predicted, target
    
from torch.optim import AdamW
from transformers import RobertaModel

def encoder(batch, model):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    batch_size, sents_per_vector, tensor_size = input_ids.shape

    # Flatten to encode all at once
    input_ids = torch.reshape(input_ids, (-1, tensor_size))
    attention_mask = torch.reshape(attention_mask, (-1, tensor_size))

    # Use [CLS] token representation
    # data augmentation handled by roberta, dropout implemented under the hood
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state[:, 0]

    # Add dropout layer for better performance?

    # Reshape back to nested tensors
    embeddings = torch.reshape(embeddings, (batch_size, sents_per_vector, -1))
    return embeddings

model = RobertaModel.from_pretrained('roberta-base')
model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 4

for epoch in range(epochs):
    total_loss = 0
    # Shape = [#features, #batch_size, #tensor_length]
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        embeddings = encoder(batch, model)
        loss = contrastive_loss(embeddings)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}")

save_directory = './largest'  # Specify your save directory
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

from datasets import load_metric

def evaluate(data_loader, model):
    y_pred, y_test = [], []
    model.eval()

    f1_metric = load_metric('f1')

    for i, batch in enumerate(data_loader):
        with torch.no_grad():
            outputs = encoder(batch, model)
            
        y_pred_batch, y_batch = contrastive_loss(outputs, train=False)
        y_test += list(y_batch.detach().numpy())
        y_pred += list(y_pred_batch.detach().numpy())

        f1_metric.add_batch(predictions=y_pred_batch, references=y_batch)
    
    return f1_metric.compute(average='macro')

model_names = ['largest']
print('Evaluating....')
for name in model_names:
    model = RobertaModel.from_pretrained(name)
    f1 = evaluate(test_dataloader, model)
    print(f'f1 for {name}: {f1}')