import sys

unwanted_path = '/mmfs1/home/hsethu/.local/lib/python3.9/site-packages'
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)


from transformers import AutoTokenizer, AutoModelForCausalLM#, DataCollatorWithPadding, HfArgumentParser, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np

def tok(examples, tokenizer, cols):
    if len(cols) == 1:
        return tokenizer(examples[cols[0]], truncation=True, padding=True, max_length=128)
    return tokenizer(examples[cols[0]], examples[cols[1]], truncation=True, padding=True, max_length=128)

def preprocess(raw_dataset, tokenizer, cols, null_lab=-1, device="cuda"):
    def tokenize_function(examples):
        tmp = tok(examples, tokenizer, cols)
        # for i in range(10):
        #     print(tokenizer.decode(tmp['input_ids'][i]))
        #     input()
        tmp['label'] = examples['label']
        for key in ['input_ids', 'attention_mask', 'label']:
            tmp[key] = np.array([inp for inp in tmp[key]])
            tmp[key] = tmp[key][np.array(tmp['label']) != null_lab]
        n = len(tmp['input_ids'])

        max_length = max(map(lambda lst: len(lst), tmp['input_ids']))
        input_ids = torch.full(((n), max_length), tokenizer.pad_token_id)
        attention_mask = torch.zeros((n), max_length)
        labels = torch.zeros(n)

        for i in range(n):
            input_ids[i, :len(tmp['input_ids'][i])] = torch.Tensor(tmp['input_ids'][i])
            attention_mask[i, :len(tmp['attention_mask'][i])] = torch.Tensor(tmp['attention_mask'][i])
            labels[i] = tmp['label'][i]

        return TensorDataset(input_ids.to(int).to(device), attention_mask.to(int).to(device), labels.to(int).to(device))


    tokenized_dataset = tokenize_function(raw_dataset)
    return tokenized_dataset