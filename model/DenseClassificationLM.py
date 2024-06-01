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

class DenseClassificationLM(nn.Module):
    def __init__(self, pretrained_model, head, num_labels, freeze_layers=False):
        super(DenseClassificationLM, self).__init__()
        self.emb_layer = pretrained_model.get_input_embeddings()
        self.pretrained_model = pretrained_model
        self.head = head
        if freeze_layers:
            for param in self.pretrained_model.parameters():
                param.required_grad = False

    def forward(self, input_ids, attention_mask, want_embs=False):
        input_embs = self.emb_layer(input_ids)
        input_embs.requires_grad_()
        input_embs.retain_grad()
        outputs = self.pretrained_model(inputs_embeds=input_embs, attention_mask=attention_mask, output_hidden_states=True)
        # outputs = self.pretrained_model(input_ids, attention_mask, output_hidden_states=True)

        # last_layer = outputs.hidden_state[-1]
        # seq_len = last_layer.shape[1]

        # sum_logits = 0
        # for i in range(seq_len):
        #     sum_logits += self.head(last_layer[:, i, ...])

        # mean_logits = sum_logits / seq_len

        logits = self.head(outputs.hidden_state[-1][:, :, ...])
        mean_logits = logits.mean(dim=1)

        if want_embs:
            return mean_logits, input_embs
        return mean_logits
