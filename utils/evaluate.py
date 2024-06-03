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
from tqdm import tqdm

def evaluate(model, loader, num_labels, classification_idx=-1, cap=1000):
    model.eval()
    # print(num_labels)
    # input()
    y_hat = []
    y = []
    with torch.no_grad():
        for batch in tqdm(loader):
            if len(y) > cap:
                break
            logits = model(batch[0], batch[1], classification_idx=classification_idx, want_embs=False)
            y_hat += logits.argmax(dim=-1).tolist()
            y += batch[2].tolist()
    avg = "binary"
    if num_labels > 2:
        avg = "macro"
    return f1_score(y, y_hat, average=avg), accuracy_score(y, y_hat), y_hat