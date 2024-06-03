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
import torch.nn.functional as F

from utils.entropy_funcs import entropy_top_k

import numpy as np

from utils.evaluate import evaluate

def masked_finetune(model, entropy_func, num_epochs, optimizer, train_loader, val_loader, num_labels, classification_idx, k=10, copy_b_size=16, device="cuda", is_llama=False):
    losses = []
    val_metrics = []
    train_metrics = []
    # print(is_llama)
    if is_llama:
        copy_b_size = 4
    
    model.train()
    for epoch in range(num_epochs):
        print(f"epoch {epoch}/{num_epochs}")
        f1, acc, _ = evaluate(model, val_loader, num_labels, classification_idx)
        val_metrics.append((f1, acc))

        train_f1, train_acc, _ = evaluate(model, train_loader, num_labels, classification_idx)
        train_metrics.append((train_f1, train_acc))
        print(f"f1: {f1}, acc: {acc}, train_f1: {train_f1}, train_acc: {train_acc}")
        pbar = tqdm(train_loader)
        for batch in pbar:
            # assume batch size of 1
            input_ids = batch[0] # (1, max_len)
            attention_mask = batch[1]
            label = batch[2]
            seq_len = torch.sum(attention_mask)

            copy_masks = attention_mask.repeat((seq_len, 1))
            input_ids_copy = input_ids.repeat((copy_b_size, 1))
            copy_masks[torch.arange(seq_len), torch.arange(seq_len)] = 0
            # print(copy_masks)
            # input() 
            probs_copy = torch.zeros(seq_len).to(device) # probability of correct class removing the ith token

            for i in range(int(seq_len/copy_b_size)): # get the probabilities if you mask out the ith word
                masks_i = copy_masks[i * copy_b_size : min((i+1) * copy_b_size, seq_len)]
                logits_i = model(input_ids_copy[:masks_i.shape[0], :], masks_i, classification_idx=classification_idx)
                # print("shape: ", F.softmax(logits_i, dim=-1).shape)
                # print("shape1: ", probs_copy[i * copy_b_size : min((i+1) * copy_b_size, seq_len)].shape)
                probs_copy[i * copy_b_size : min((i+1) * copy_b_size, seq_len)] = torch.squeeze(F.softmax(logits_i, dim=-1)[..., label])

            # compute the entropy
            output = model(batch[0], batch[1], classification_idx=classification_idx)
            prob = F.softmax(output, dim=-1)[..., label]
            diffs = torch.absolute(prob - probs_copy)# F.softmax(torch.topk(torch.absolute(prob - probs_copy), k).values)
            # if k >= len(probs_copy): 
            #     print(len(probs_copy), k, input_ids)
            #     input()
            entropy = entropy_top_k(diffs, min(k, seq_len - 1))

            # now compute the actual loss
            loss = entropy_func(F.cross_entropy(output, label), entropy)
            pbar.set_description(f"Loss: {round(loss.item(), 5)}")
            optimizer.zero_grad()
            loss.backward()

            if torch.sum(torch.isnan(output)) > 0:
                print("ERROR")
                return None
            
            # update
            optimizer.step()
            losses.append(loss.item())

    post_f1, post_acc, post_y_hat = evaluate(model, val_loader, num_labels, classification_idx)
    train_post_f1, train_post_acc, train_post_y_hat = evaluate(model, train_loader, num_labels, classification_idx)

    val_metrics.append((post_f1, post_acc))
    train_metrics.append((train_post_f1, train_post_acc))
    print(f"post_f1:{post_f1}, post_acc: {post_acc}")
    return train_metrics, val_metrics, post_y_hat, losses