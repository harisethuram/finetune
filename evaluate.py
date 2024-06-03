import argparse, sys

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
from utils.evaluate import evaluate

from model.ClassificationHead import ClassificationHead
from model.ClassificationLM import ClassificationLM

from keys import HF_TOKEN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NULL_LAB = -1
CLASSIFICATION_IDX = -1
B_SIZE = 16
NUM_LABELS = 3
VAL_CAP = 1000
parser=argparse.ArgumentParser()
parser.add_argument("--finetuned_model_dir")
parser.add_argument("--model_name")
parser.add_argument("--dataset_cache")
parser.add_argument("--results_dir")
parser.add_argument("--val_dataset")
parser.add_argument("--val_data_dir")
parser.add_argument("--split")
parser.add_argument("--num_labels", type=int, default=2)
parser.add_argument("--null_lab", type=int, default=-1)
parser.add_argument("--b_size", type=int, default=16)
parser.add_argument("--classification_idx", type=int, default=-1)
parser.add_argument("--cols")
args = parser.parse_args()
print(args)

def main():
    print("OK")
    if not os.path.exists(args.finetuned_model_dir):
        print("model does not exist!")
        return None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(args.finetuned_model_dir).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    args.cols = args.cols.split(",")

    print(F"Loading {args.val_dataset}/{args.val_data_dir}...")
    raw_dataset = load_dataset(args.val_dataset, args.val_data_dir, split=args.split, cache_dir=args.dataset_cache)
    # print(raw_dataset.keys())
    # input()
    

    # train_dataset = raw_dataset['train'][:TRAIN_CAP]
    # test_dataset = raw_dataset['test'][:TEST_CAP]

    print("Preprocessing data...")
    validation_loader = DataLoader(preprocess(raw_dataset, tokenizer, args.cols, args.null_lab, DEVICE), args.b_size)

    print("evaluating...")
    f1, acc, y_hat = evaluate(model, validation_loader, args.num_labels, args.classification_idx)
    print("Saving...")
    os.makedirs(os.path.dirname(args.results_dir), exist_ok=True)
    # os.makedirs(os.path.dirname(FINETUNED_MODEL_DIR), exist_ok=True)  

    with open(args.results_dir+"val_results.csv", "w+") as f:
        f.write("f1,acc\n")
        f.write(f"{f1},{acc}")
        
    with open(args.results_dir+"post_preds.txt", "w+") as f:
        f.write("\n".join([f"{ps}\t{true}" for ps, true in zip(y_hat, raw_dataset["label"])]))
    

    print("Complete!")
if __name__ == "__main__":  
    main()