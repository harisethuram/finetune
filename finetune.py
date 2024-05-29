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

from utils.preprocess import preprocess
from utils.finetune import finetune
from utils.masked_finetune import masked_finetune
from utils.custom_loss import CompetentLoss, CrossEntropyLoss
from utils.entropy_funcs import EntropyFunc

from model.ClassificationHead import ClassificationHead
from model.ClassificationLM import ClassificationLM

from keys import HF_TOKEN

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="model name")
parser.add_argument("--model_cache", help="model cache dir")
parser.add_argument("--dataset", help="dataset name")
parser.add_argument("--data_dir")
parser.add_argument("--dataset_cache")
parser.add_argument("--results_dir")
parser.add_argument("--num_epochs", type=int)
parser.add_argument("--null_lab", type=int)
parser.add_argument("--classification_idx", type=int)
parser.add_argument("--train_b_size", type=int)
parser.add_argument("--val_b_size", type=int)
parser.add_argument("--num_labels", type=int)
parser.add_argument("--train_cap", type=int, nargs='?', default=1000)
parser.add_argument("--val_cap", type=int, nargs='?', default=1000)
parser.add_argument("--test_cap", type=int, nargs='?', default=1000)
parser.add_argument("--input_cols", type=lambda x: x.split(","))
parser.add_argument("--loss_fn_name")
parser.add_argument("--entropy_func")
parser.add_argument("--lambda_entropy", type=float, default=0.1)
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--copy_b_size", type=int, default=16, help="For the masking, batch size to process")
parser.add_argument("--overwrite", action="store_true")

args = parser.parse_args()
print(args)

def main():
    try:
        with open(args.results_dir + "losses.txt") as fresults_exist:
            if len(fresults_exist.readlines()) > 0 and not args.overwrite:
                print(f'Results already exists. exiting.')
                return
    except Exception as e:
        pass 

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {DEVICE}")
    print(f"train batch size: {args.train_b_size}, val batch size: {args.val_b_size}")
    print(f"Loading {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.model_cache, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("pad_id:", tokenizer.pad_token_id)
    print(model.config.hidden_size, args.num_labels)
    head = ClassificationHead(model.config.hidden_size, args.num_labels)

    classification_model = ClassificationLM(model, head, args.num_labels)
    classification_model.to(DEVICE)
    classification_model.half()
    param_dtype = next(classification_model.parameters()).dtype
    print("Data type of the model parameters:", param_dtype)

    
    print(F"Loading {args.dataset}/{args.data_dir}...")
    raw_dataset = load_dataset(args.dataset, args.data_dir, cache_dir=args.dataset_cache)

    train_dataset = raw_dataset['train'][:args.train_cap]
    validation_dataset = raw_dataset['validation'][:args.val_cap]

    # currently, MASK only supports b_size=1
    if args.loss_fn_name == "MASK": args.train_b_size = 1 

    print("Preprocessing data...")
    train_loader = DataLoader(preprocess(train_dataset, tokenizer, args.input_cols, args.null_lab, DEVICE), args.train_b_size)
    validation_loader = DataLoader(preprocess(validation_dataset, tokenizer, args.input_cols, args.null_lab, DEVICE), args.val_b_size)

    # Get the entropy function
    entropy_func = EntropyFunc(args.entropy_func, args.lambda_entropy)

    print("Finetuning...")
    optimizer = optim.SGD(classification_model.parameters(), lr=0.0001)#, momentum=0.3, nesterov=True)
    if args.loss_fn_name == "MASK":
        train_metrics, val_metrics, post_y_hat, losses = masked_finetune(classification_model, entropy_func, args.num_epochs, optimizer,\
                                                                         train_loader, validation_loader, args.num_labels,\
                                                                         args.classification_idx, args.k, args.copy_b_size, DEVICE)
    else:
        if args.loss_fn_name == "CE":
            loss_fn = CrossEntropyLoss()
        elif args.loss_fn_name == "COMP": 
            loss_fn = CompetentLoss()
        else:
            raise Exception("Incorrect loss_fn name, either \"CE\" or \"COMP\"")
        train_metrics, val_metrics, post_y_hat, losses = finetune(classification_model, entropy_func, loss_fn, args.num_epochs, optimizer,\
                                                                  train_loader, validation_loader, args.num_labels, args.classification_idx)
    
    print("Saving...")
    os.makedirs(os.path.dirname(args.results_dir), exist_ok=True)
    os.makedirs(os.path.dirname(args.finetuned_model_dir), exist_ok=True)

    with open(args.results_dir+"val_results.csv", "w+") as f:
        f.write("f1,acc\n")
        for f1, acc in val_metrics:
            f.write(f"{f1},{acc}\n")
    
    with open(args.results_dir+"train_results.csv", "w+") as f:
        f.write("f1,acc\n")
        for f1, acc in train_metrics:
            f.write(f"{f1},{acc}\n")
        
    with open(args.results_dir+"losses.txt", "w+") as f:
        f.write("\n".join([str(loss) for loss in losses]))

    plt.plot(range(len(losses)), losses)
    plt.title(f"Losses: {args.model_name}, {args.dataset}/{args.data_dir}")
    plt.ylabel("loss")
    plt.xlabel("step")
    plt.savefig(args.results_dir+"loss.png")

    with open(args.results_dir+"post_preds.txt", "w+") as f:
        f.write("\n".join([str(ps) for ps in post_y_hat]))
    
    torch.save(classification_model, args.results_dir+"model.pt")
    print("Complete!")
    return

if __name__ == "__main__": 
    main()
