#!/usr/bin/env python3
'''''''''''''''''''''''
COPYLEFT LESTERRRY, 2020
'''''''''''''''''''''''
#This script generates text based on 'model.bin' file, retrieved from learning.
#Make sure file is located in script directory and is readable.
#Serve -o argument to simply print predicrion
#Serve -p argument if picture is needed
import sys
import os
import pickle
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pic #Comment this if you're not going to generate pics

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

#Replace this with 'CtI' dict retrieved while learning
CTI = {' ': 0, 'о': 1, 'а': 2, 'е': 3, 'т': 4, 'и': 5, 'н': 6, 'с': 7, 'л': 8, 'р': 9, 'в': 10, 'к': 11, 'м': 12, 'д': 13, 'у': 14, 'п': 15, 'ь': 16, 'я': 17, ',': 18, 'ы': 19, 'б': 20,$
#Replace this with 'ItC' dict retrieved while learning
ITC = {0: ' ', 1: 'о', 2: 'а', 3: 'е', 4: 'т', 5: 'и', 6: 'н', 7: 'с', 8: 'л', 9: 'р', 10: 'в', 11: 'к', 12: 'м', 13: 'д', 14: 'у', 15: 'п', 16: 'ь', 17: 'я', 18: ',', 19: 'ы', 20: 'б',$

#Replace this with prediction length you want
PRED_LEN = 130

#Enter file path here
file = open(os.path.join(__location__, 'model55.bin'), "rb")

class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)
    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)
    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
               torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))

def evaluate(model, char_to_idx, idx_to_char, start_text=' ', prediction_len=200, temp=0.3):
    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    _, hidden = model(train, hidden)

    inp = train[-1].view(-1, 1, 1)
    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char
    return predicted_text

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
mod = torch.load(file, map_location=device)
mod.eval()

text = (evaluate(
        mod,
        CTI,
        ITC,
        temp=0.3,
        prediction_len=PRED_LEN,
        start_text=' '
        )
    )
text = text.split(' ')
text = text[1:-1]
text = ' '.join(text)
text = text.lower().capitalize()
if '-o' in sys.argv:
    print(text)
elif '-p' in sys.argv:
    pic.draw(text)
