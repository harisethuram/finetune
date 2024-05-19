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

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(0.1)  # Adjust dropout rate as needed
        self.linear1 = nn.Linear(hidden_size, hidden_size // 2)  # Reduce dimension
        self.relu = nn.ReLU()  # ReLU activation function
        self.linear2 = nn.Linear(hidden_size // 2, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Optional: additional dropout for regularization
        x = self.linear2(x)
        return x