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
CTI = {' ': 0, 'о': 1, 'а': 2, 'е': 3, 'т': 4, 'и': 5, 'н': 6, 'с': 7, 'л': 8, 'р': 9, 'в': 10, 'к': 11, 'м': 12, 'д': 13, 'у': 14, 'п': 15, 'ь': 16, 'я': 17, ',': 18, 'ы': 19, 'б': 20, 'ч': 21, '.': 22, 'й': 23, 'г': 24, 'з': 25, 'ш': 26, 'ж': 27, 'х': 28, 'ю': 29, '?': 30, '-': 31, '6': 32, 'А': 33, '1': 34, 'ц': 35, 'К': 36, 'С': 37, 'Н': 38, 'э': 39, 'В': 40, 'О': 41, 'П': 42, '8': 43, 'Е': 44, 'Т': 45, ':': 46, 'щ': 47, 'И': 48, 'М': 49, '9': 50, '0': 51, 'Д': 52, 'ё': 53, 'ф': 54, 'Л': 55, '!': 56, '7': 57, 'Р': 58, '3': 59, '2': 60, 'Я': 61, 'У': 62, 'Г': 63, 'Б': 64, '5': 65, '4': 66, 'Ч': 67, 'З': 68, 'Х': 69, 'Ш': 70, 'Ь': 71, 'Ы': 72, 'Э': 73, 'Ж': 74, 'Й': 75, 'ъ': 76, 'Ф': 77, 'Ю': 78, 'Ц': 79, 'Щ': 80, 'Ё': 81, 'ј': 82, 'Ξ': 83, 'Ъ': 84, 'ي': 85, 'ن': 86, 'م': 87, 'ͅ': 88, 'r': 89, 'љ': 90, 'ヽ': 91, 's': 92, 'p': 93, 'ل': 94, 'n': 95, 'a': 96, 'ㄛ': 97, 'ㄗ': 98, 'ㄙ': 99, 'ر': 100, 'ا': 101, 'ف': 102, 'ه': 103, 'ཀ': 104, 'ʖ': 105, 'C': 106, 'イ': 107, 'ﾉ': 108, 'Ｙ': 109, 'ー': 110, 'ج': 111, 'ى': 112, 'إ': 113, 'ش': 114, 'ء': 115, 'ت': 116, 'ع': 117, 'ك': 118, 'أ': 119, 'ؤ': 120, 'س': 121, 'π': 122, 'ż': 123, 'њ': 124, 'џ': 125, '𓂺': 126, 'Y': 127, 'L': 128, 'O': 129, 'N': 130, 'E': 131, 'D': 132, 'e': 133, 'k': 134, 't': 135, 'o': 136, 'ノ': 137, 'ト': 138, '仝': 139, 'ミ': 140, '土': 141, '彡': 142, 'Ѽ': 143, '\n': 144}
#Replace this with 'ItC' dict retrieved while learning
ITC = {0: ' ', 1: 'о', 2: 'а', 3: 'е', 4: 'т', 5: 'и', 6: 'н', 7: 'с', 8: 'л', 9: 'р', 10: 'в', 11: 'к', 12: 'м', 13: 'д', 14: 'у', 15: 'п', 16: 'ь', 17: 'я', 18: ',', 19: 'ы', 20: 'б', 21: 'ч', 22: '.', 23: 'й', 24: 'г', 25: 'з', 26: 'ш', 27: 'ж', 28: 'х', 29: 'ю', 30: '?', 31: '-', 32: '6', 33: 'А', 34: '1', 35: 'ц', 36: 'К', 37: 'С', 38: 'Н', 39: 'э', 40: 'В', 41: 'О', 42: 'П', 43: '8', 44: 'Е', 45: 'Т', 46: ':', 47: 'щ', 48: 'И', 49: 'М', 50: '9', 51: '0', 52: 'Д', 53: 'ё', 54: 'ф', 55: 'Л', 56: '!', 57: '7', 58: 'Р', 59: '3', 60: '2', 61: 'Я', 62: 'У', 63: 'Г', 64: 'Б', 65: '5', 66: '4', 67: 'Ч', 68: 'З', 69: 'Х', 70: 'Ш', 71: 'Ь', 72: 'Ы', 73: 'Э', 74: 'Ж', 75: 'Й', 76: 'ъ', 77: 'Ф', 78: 'Ю', 79: 'Ц', 80: 'Щ', 81: 'Ё', 82: 'ј', 83: 'Ξ', 84: 'Ъ', 85: 'ي', 86: 'ن', 87: 'م', 88: 'ͅ', 89: 'r', 90: 'љ', 91: 'ヽ', 92: 's', 93: 'p', 94: 'ل', 95: 'n', 96: 'a', 97: 'ㄛ', 98: 'ㄗ', 99: 'ㄙ', 100: 'ر', 101: 'ا', 102: 'ف', 103: 'ه', 104: 'ཀ', 105: 'ʖ', 106: 'C', 107: 'イ', 108: 'ﾉ', 109: 'Ｙ', 110: 'ー', 111: 'ج', 112: 'ى', 113: 'إ', 114: 'ش', 115: 'ء', 116: 'ت', 117: 'ع', 118: 'ك', 119: 'أ', 120: 'ؤ', 121: 'س', 122: 'π', 123: 'ż', 124: 'њ', 125: 'џ', 126: '𓂺', 127: 'Y', 128: 'L', 129: 'O', 130: 'N', 131: 'E', 132: 'D', 133: 'e', 134: 'k', 135: 't', 136: 'o', 137: 'ノ', 138: 'ト', 139: '仝', 140: 'ミ', 141: '土', 142: '彡', 143: 'Ѽ', 144: '\n'}

#Replace this with prediction length you want
PRED_LEN = 130

file = open(os.path.join(__location__, 'model1000c.bin'), "rb")

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

def textev():
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
    return text
if '-o' in sys.argv:
    print(textev())
elif '-p' in sys.argv:
    if '-c' in sys.argv:
        i = 0
        while(True):
            #print(i)
            i += 1
            #text = (evaluate(mod, CTI, ITC, temp=0.3, prediction_len=PRED_LEN, start_text=' '))
            t = textev()
            print("\r>>", i, t, end='')
            pic.draw(textev())
    else:
        pic.draw(textev())
