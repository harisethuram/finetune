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

from utils.preprocess import preprocess
from utils.finetune import finetune
from utils.custom_loss import CompetentLoss

from model.ClassificationHead import ClassificationHead
from model.ClassificationLM import ClassificationLM

import argparse

parser = argparse.ArgumentParser()

TOKEN = "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"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = sys.argv[1]
MODEL_CACHE = sys.argv[2]
DATASET = sys.argv[3]
DATA_DIR = sys.argv[4]
DATASET_CACHE = sys.argv[5]
RESULTS_DIR = sys.argv[6]
FINETUNED_MODEL_DIR = sys.argv[7]
NUM_EPOCHS = int(sys.argv[8])
NULL_LAB = int(sys.argv[9])
CLASSIFICATION_IDX = int(sys.argv[10])
B_SIZE = int(sys.argv[11])
NUM_LABELS = int(sys.argv[12])
TRAIN_CAP = int(sys.argv[13])
VAL_CAP = int(sys.argv[14])
TEST_CAP = int(sys.argv[15])
cols = sys.argv[16].split(",")
    

def main():
    print(f"batch size: {B_SIZE}")
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE, token=TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("pad_id:", tokenizer.pad_token_id)
    print(model.config.hidden_size, NUM_LABELS)
    head = ClassificationHead(model.config.hidden_size, NUM_LABELS)

    classification_model = ClassificationLM(model, head, NUM_LABELS)
    classification_model.to(DEVICE)
    classification_model.half()
    param_dtype = next(classification_model.parameters()).dtype
    print("Data type of the model parameters:", param_dtype)

    
    print(F"Loading {DATASET}/{DATA_DIR}...")
    raw_dataset = load_dataset(DATASET, DATA_DIR, cache_dir=DATASET_CACHE)

    train_dataset = raw_dataset['train'][:TRAIN_CAP]
    validation_dataset = raw_dataset['validation'][:VAL_CAP]
    test_dataset = raw_dataset['test'][:TEST_CAP]

    print("Preprocessing data...")
    train_loader = DataLoader(preprocess(train_dataset, tokenizer, cols, NULL_LAB, DEVICE), B_SIZE)
    validation_loader = DataLoader(preprocess(validation_dataset, tokenizer, cols, NULL_LAB, DEVICE), B_SIZE)
    test_loader = DataLoader(preprocess(test_dataset, tokenizer, cols, NULL_LAB, DEVICE), B_SIZE)

    print("Finetuning...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classification_model.parameters(), lr=0.0001)#, momentum=0.3, nesterov=True)
    train_metrics, val_metrics, post_y_hat, losses = finetune(classification_model, loss_fn, NUM_EPOCHS, optimizer, train_loader, validation_loader, NUM_LABELS, CLASSIFICATION_IDX)
    print("Saving...")
    os.makedirs(os.path.dirname(RESULTS_DIR), exist_ok=True)
    os.makedirs(os.path.dirname(FINETUNED_MODEL_DIR), exist_ok=True)  

    with open(RESULTS_DIR+"val_results.csv", "w+") as f:
        f.write("f1,acc\n")
        for f1, acc in val_metrics:
            f.write(f"{f1},{acc}\n")
    
    with open(RESULTS_DIR+"train_results.csv", "w+") as f:
        f.write("f1,acc\n")
        for f1, acc in train_metrics:
            f.write(f"{f1},{acc}\n")
        
    with open(RESULTS_DIR+"losses.txt", "w+") as f:
        f.write("\n".join([str(loss) for loss in losses]))

    plt.plot(range(len(losses)), losses)
    plt.title(f"Losses: {MODEL_NAME}, {DATASET}/{DATA_DIR}")
    plt.ylabel("loss")
    plt.xlabel("step")
    plt.savefig(RESULTS_DIR+"loss.png")

    with open(RESULTS_DIR+"post_preds.txt", "w+") as f:
        f.write("\n".join([str(ps) for ps in post_y_hat]))
    
    torch.save(classification_model, FINETUNED_MODEL_DIR)
    print("Complete!")
    
main()