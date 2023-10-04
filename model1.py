# code for Glove word embedding
from sklearn.metrics import f1_score
import numpy as np
import math
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# def embedding_for_vocab(filepath, word_index,
# 						embedding_dim):
# 	vocab_size = len(word_index) + 1
	
# 	# Adding again 1 because of reserved 0 index
# 	embedding_matrix_vocab = np.zeros((vocab_size,
# 									embedding_dim))

# 	with open(filepath, "r",encoding="utf8" ) as f:
# 		for line in f:
# 			word,*vector = line.split()
# 			if word in word_index:
# 				idx = word_index[word]
# 				embedding_matrix_vocab[idx] = np.array(
# 					vector, dtype=np.float32)[:embedding_dim]

# 	return embedding_matrix_vocab

def map_label(label):
    # print(label)
    if(label == "positive"):
        return 2
    elif(label == "neutral"):
        return 1
    else:
        return 0

def get_train_data():
    df = pd.read_excel("/data/home/jaydeeps/DLNLP/Assignment1/datasets/Datasets-Splits/Assignment 1/ClassificationDataset-train0.xlsx", header=None)
    # print(df.head())
    # df.iloc[:1] = df.iloc[:1].apply(map_label)
    df[0] = df[0].apply(map_label)
    data_list = []
    for index, row in df.iterrows():
        label = row[0]
        text = row[1]
        data_list.append([label, text])

    print(len(data_list))
    print(data_list[0], data_list[1])
    return data_list
    # return xlsx.reader(open("/data/home/jaydeeps/DLNLP/Assignment1/datasets/Datasets-Splits/Assignment 1/ClassificationDataset-train0.xlsx", "r"))

def get_validate_data():
    df = pd.read_excel("/data/home/jaydeeps/DLNLP/Assignment1/datasets/Datasets-Splits/Assignment 1/ClassificationDataset-valid0.xlsx", header=None)
    # df.iloc[:1] = df.iloc[:1].apply(map_label)
    df[0] = df[0].apply(map_label)
    data_list = []
    for index, row in df.iterrows():
        label = row[0]
        text = row[1]
        data_list.append([label, text])

    # print(len(data_list))
    # print(data_list[0], data_list[1])
    return data_list
    # return df
    # return xlsx.reader(open("/data/home/jaydeeps/DLNLP/Assignment1/datasets/Datasets-Splits/Assignment 1/ClassificationDataset-valid0.xlsx", "r"))

def preprocess(text):
    # separate punctuations
    text = text.replace(".", " . ") \
                 .replace(",", " , ") \
                 .replace(";", " ; ") \
                 .replace("?", " ? ")
    return text.split()

embedding_dim = 50
glove = torchtext.vocab.GloVe(name="840B", dim=300, max_vectors=20000)
def get_train_vectors():
    train = []
    for i, line in enumerate(get_train_data()):
        text = line[-1]
        # print("text:: ", text)
        # tokenizer = Tokenizer()
        # tokenizer.fit_on_texts(preprocess(text))
        # print("*******", tokenizer.word_index)
        
        # embedding_matrix_vocab = embedding_for_vocab('glove.840B.300d.txt', tokenizer.word_index, embedding_dim)
		
        vector_sum = sum(glove[w] for w in preprocess(text))
        # vector_avg = sum(glove[w] for w in preprocess(text)) / len(preprocess(text))
  
        # print("************")
        # print(line[0])
        # print(line[1])
        # print(line[2])
        label = torch.tensor(int(line[0])).long()
		
        train.append((vector_sum, label))
    return train

def get_validation_vectors():
    validate = []
    for i, line in enumerate(get_validate_data()):
        text = line[-1]
        # tokenizer = Tokenizer()
        # tokenizer.fit_on_texts(preprocess(text))
        # embedding_matrix_vocab = embedding_for_vocab('glove.840B.300d.txt', tokenizer.word_index, embedding_dim)
		
        vector_sum = sum(glove[w] for w in preprocess(text))
        # vector_avg = sum(glove[w] for w in preprocess(text)) / len(preprocess(text))
        label = torch.tensor(int(line[0])).long()
		
        validate.append((vector_sum, label))
    return validate

print("#######", "imports were SUCCESS", "##########")
train = get_train_vectors()
valid = get_validation_vectors()
train_loader = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=100, shuffle=True)

def train_network(model, train_loader, valid_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    for epoch in range(num_epochs):
        for text, labels in train_loader:
            optimizer.zero_grad()
            pred = model(text)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        losses.append(float(loss))     
        if epoch % 5 == 4:
            epochs.append(epoch)
            train_acc.append(get_accuracy(model, train_loader))
            valid_acc.append(get_accuracy(model, valid_loader))
            print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
                epoch+1, loss, train_acc[-1], valid_acc[-1]))

    # plotting
    plt.title("Training Curve")
    plt.plot(losses, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plt.savefig("Losses1.png")

    plt.clf()
    
    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, valid_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    plt.savefig("Accuracy1.png")

def get_accuracy(model, data_loader):
    correct, total = 0, 0
    for text, labels in data_loader:
        output = model(text)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
    return correct / total

def get_f1_score(model, data_loader):
    y_true = []
    y_pred = []
    for text, labels in data_loader:
        output = model(text)
        pred = output.max(1, keepdim=True)[1]
        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist())
    return f1_score(y_true, y_pred, average='micro')  # You can change the averaging method if needed

mymodel = nn.Sequential(nn.Linear(300, 200),
                        nn.ReLU(),
                        nn.Linear(200, 100),
                        nn.ReLU(),
                        nn.Linear(100, 30),
                        nn.ReLU(),
                        nn.Linear(30, 3))
train_network(mymodel, train_loader, valid_loader, num_epochs=40, learning_rate=1e-4)
accuracy = get_accuracy(mymodel, valid_loader)
print("##############", "SUCCESS", "#################")
print("validation Accuracy", accuracy)

print("***********************")
f1_avg = get_f1_score(mymodel, valid_loader)
print("Average F1 Score:", f1_avg)