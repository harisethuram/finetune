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

class ClassificationLM(nn.Module):
    def __init__(self, pretrained_model, head, num_labels, freeze_layers=False):
        super(ClassificationLM, self).__init__()
        self.emb_layer = pretrained_model.get_input_embeddings()
        self.pretrained_model = pretrained_model
        self.head = head
        if freeze_layers:
            for param in self.pretrained_model.parameters():
                param.required_grad = False
        
    def forward(self, input_ids, attention_mask, classification_idx=-1):
        input_embs = self.emb_layer(input_ids)
        outputs = self.pretrained_model(inputs_embeds=input_embs, attention_mask=attention_mask, output_hidden_states=True)
        # outputs = self.pretrained_model(input_ids, attention_mask, output_hidden_states=True)
        logits = self.head(outputs.hidden_states[-1][:, classification_idx, ...])
        return logits, input_embs