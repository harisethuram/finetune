import sys

unwanted_path = '/mmfs1/home/hsethu/.local/lib/python3.9/site-packages'
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)


from transformers import AutoTokenizer, AutoModelForCausalLM
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

from utils.evaluate import evaluate

def finetune(model, entropy_func, loss_fn, num_epochs, optimizer, train_loader, val_loader, num_labels, classification_idx):
    losses = []
    val_metrics = []
    train_metrics = []
    
    model.train()
    # print(num_labels)
    # input()

    for epoch in range(num_epochs):
        print(f"epoch {epoch}/{num_epochs}")
        f1, acc, _ = evaluate(model, val_loader, num_labels, classification_idx)
        val_metrics.append((f1, acc))

        train_f1, train_acc, _ = evaluate(model, train_loader, num_labels, classification_idx)
        train_metrics.append((train_f1, train_acc))
        print(f"f1: {f1}, acc: {acc}, train_f1: {train_f1}, train_acc: {train_acc}")
        pbar = tqdm(train_loader)
        for batch in pbar:
            logits, embs = model(batch[0], batch[1], classification_idx=classification_idx, want_embs=True)
            loss = loss_fn(logits, batch[2], embs, entropy_func)
            pbar.set_description(f"Loss: {round(loss.item(), 5)}")
            optimizer.zero_grad()
            loss.backward()
            if torch.sum(torch.isnan(logits)) > 0:
                print("ERROR")
                return None

            optimizer.step()
            losses.append(loss.item())

    post_f1, post_acc, post_y_hat = evaluate(model, val_loader, num_labels, classification_idx)
    train_post_f1, train_post_acc, train_post_y_hat = evaluate(model, train_loader, num_labels, classification_idx)

    val_metrics.append((post_f1, post_acc))
    train_metrics.append((train_post_f1, train_post_acc))
    print(f"post_f1:{post_f1}, post_acc: {post_acc}")
    return train_metrics, val_metrics, post_y_hat, losses