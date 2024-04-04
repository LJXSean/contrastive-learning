import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# load the data
dataset = pd.read_csv('train.csv')
kaggle = pd.read_csv('test.csv')

def preprocess(sentence):
    sentence = sentence.strip().lower()
    sentence = re.sub(r'[^\w\s!?"\']', '', sentence)

    tokens = word_tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

X = dataset['Text']#.apply(lambda x: preprocess(x['Text']), axis=1, result_type='expand')
K = kaggle['Text']#.apply(lambda x: preprocess(x['Text']), axis=1, result_type='expand')
y = dataset['Verdict'] + 1

# Split sentences and labels into training and test set with a test set size of 20%
sentences_train, sentences_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Size of training set: {}".format(len(sentences_train)))
print("Size of test set: {}".format(len(sentences_test)))

# Create Term-Document Matrix for different n-gram sizes
#vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
vectorizer = TfidfVectorizer(max_features=2472,norm='l1',ngram_range=(1,1),stop_words='english',strip_accents='ascii',analyzer='word')
#vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=10000)

y_train = np.asarray(labels_train, dtype=np.int8)
y_test = np.asarray(labels_test, dtype=np.int8)

# Vectorize both training and test set
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)
K_test = vectorizer.transform(K)

# The default Tensor stores float values
X_train = torch.Tensor(X_train.todense())
X_test = torch.Tensor(X_test.todense())
K_test = torch.Tensor(K_test.todense())

# Our labels are integers, hence we use LongTensor
# (that's required, otherwise we would get an error later)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(K_test.shape)

dataset_train = TensorDataset(X_train, y_train)
dataset_test = TensorDataset(X_test, y_test)
dataset_k = TensorDataset(K_test, K_test[:, 0])

batch_size = 64

loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
loader_k = DataLoader(dataset_k, batch_size=batch_size, shuffle=True)

class SimpleNet2(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.net = nn.Sequential(
            nn.Linear(self.vocab_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, X):
        log_probs = self.net(X)
        return log_probs
    
# Create the model and move to device
classifier = SimpleNet2(X_train.shape[1])

print(classifier)

def evaluate(model, loader):

    # Set model to "eval" mode (not needed here, but a good practice)
    model.eval()

    # Collect predictions and ground truth for all samples across all batches
    y_pred, y_test = [], []

        
    # Loop over each batch in the data loader
    for X_batch, y_batch in loader:
        # Push batch through network to get log probabilities for each sample in batch
        log_probs = model(X_batch)                

        # The predicted labels are the index of the higest log probability (for each sample)
        y_batch_pred = torch.argmax(log_probs, dim=1)

        # Add predictions and ground truth for current batch
        y_test += list(y_batch.detach().numpy())
        y_pred += list(y_batch_pred.detach().numpy())

    # Set model to "train" mode (not needed here, but a good practice)
    model.train()            
            
    # Return the f1 score as the output result
    return metrics.f1_score(y_test, y_pred, average='macro')

def train(model, loader_train, loader_test, optimizer, criterion, num_epochs):

    losses, f1_train, f1_test = [], [], []
    
    # Set model to "train" mode (not needed here, but a good practice)
    model.train()

    # Run all epochs
    for epoch in range(1, num_epochs+1):

        # Initialize epoch loss (cummulative loss fo all batchs)
        epoch_loss = 0.0

        # Loop over each batch in the data loader
        for X_batch, y_batch in loader_train:
            # Push batch through network to get log probabilities for each sample in batch
            log_probs = classifier(X_batch)                

            # Calculate loss
            loss = criterion(log_probs, y_batch)

            ### PyTorch Magic! ###
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Keep track of overall epoch loss
            epoch_loss += loss.item()

        
        # Keep track of all epoch losses
        losses.append(epoch_loss)
        
        # Compute f1 score for both TRAINING and TEST data
        f1_tr = evaluate(model, loader_train)
        f1_te = evaluate(model, loader_test)
        f1_train.append(f1_tr)
        f1_test.append(f1_te)

        print("Loss:\t{:.3f}, f1 train: {:.3f}, f1 test: {:.3f} (epoch {})".format(epoch_loss, f1_tr, f1_te, epoch))
     
    # Return all losses and f1 scores (all = for each epoch)
    return losses, f1_train, f1_test        

# Create the model and movie to device
classifier = SimpleNet2(X_train.shape[1])

# Define loss function
criterion = nn.NLLLoss()

# Define optimizer (you can try, but the basic (Stochastic) Gradient Descent is actually not great)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 25

losses, f1_train, f1_test = train(classifier, loader_train, loader_test, optimizer, criterion, num_epochs)

def predict(model, loader):

    # Set model to "eval" mode (not needed here, but a good practice)
    model.eval()

    # Collect predictions and ground truth for all samples across all batches
    y_pred = []

    # Loop over each batch in the data loader
    for X_batch, y_batch in loader:        
        with torch.no_grad():
            log_probs = model(X_batch)                
            y_batch_pred = torch.argmax(log_probs, dim=1)
            y_pred += list(y_batch_pred.detach().numpy())

    model.train()            
    return y_pred

result = predict(classifier, loader_k)
kaggle['Verdict'] = pd.Series(result) - 1
kaggle.drop(columns=['Text'], inplace=True)
kaggle.to_csv('A0233694X.csv', index=False)



