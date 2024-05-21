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
from utils.evaluate import evaluate

from model.ClassificationHead import ClassificationHead
from model.ClassificationLM import ClassificationLM

from keys import HF_TOKEN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_NAME = sys.argv[1]
# MODEL_CACHE = sys.argv[2]
# DATASET = sys.argv[3]
# DATA_DIR = sys.argv[4]
# DATASET_CACHE = sys.argv[5]
# RESULTS_DIR = sys.argv[6]
# FINETUNED_MODEL_DIR = sys.argv[7]
# NUM_EPOCHS = int(sys.argv[8])
NULL_LAB = -1
CLASSIFICATION_IDX = -1
B_SIZE = 16
NUM_LABELS = 3
VAL_CAP = 1000
cols = "premise,hypothesis".split(",")
    

def main():
    print(f"batch size: {B_SIZE}")
    TOKEN = "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"
    DATASET_CACHE="/gscratch/ark/hari/generative-classification/generative-classification/models/datasets/"

    tr_dataset = "stanfordnlp/snli"
    tr_dir = "plain_text"
    model_name = "meta-llama/Llama-2-7b-hf"
    tr_path = f"/gscratch/ark/hari/msc/results/{tr_dataset}/{tr_dir}/{model_name}/"
    model = torch.load(tr_path + "model.pt").cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = "adv_glue"
    data_dir = "adv_mnli"
    path = f"/gscratch/ark/hari/msc/results/eval/{dataset}/{data_dir}/{model_name}/"
    print(F"Loading {dataset}/{data_dir}...")
    raw_dataset = load_dataset(dataset, data_dir, cache_dir=DATASET_CACHE)

    # train_dataset = raw_dataset['train'][:TRAIN_CAP]
    validation_dataset = raw_dataset['validation'][:VAL_CAP]
    # test_dataset = raw_dataset['test'][:TEST_CAP]

    print("Preprocessing data...")
    validation_loader = DataLoader(preprocess(validation_dataset, tokenizer, cols, NULL_LAB, DEVICE), B_SIZE)

    print("evaluating...")
    f1, acc, y_hat = evaluate(model, validation_loader, NUM_LABELS, CLASSIFICATION_IDX)
    print("Saving...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # os.makedirs(os.path.dirname(FINETUNED_MODEL_DIR), exist_ok=True)  

    with open(path+"val_results.csv", "w+") as f:
        f.write("f1,acc\n")
        f.write(f"{f1},{acc}")
        
    with open(path+"post_preds.txt", "w+") as f:
        f.write("\n".join([str(ps) for ps in y_hat]))
    
    # with open(RESULTS_DIR+"train_results.csv", "w+") as f:
    #     f.write("f1,acc\n")
    #     for f1, acc in train_metrics:
    #         f.write(f"{f1},{acc}\n")
        
    # with open(RESULTS_DIR+"losses.txt", "w+") as f:
    #     f.write("\n".join([str(loss) for loss in losses]))

    # plt.plot(range(len(losses)), losses)
    # plt.title(f"Losses: {MODEL_NAME}, {DATASET}/{DATA_DIR}")
    # plt.ylabel("loss")
    # plt.xlabel("step")
    # plt.savefig(RESULTS_DIR+"loss.png")

    # with open(RESULTS_DIR+"post_preds.txt", "w+") as f:
    #     f.write("\n".join([str(ps) for ps in post_y_hat]))
    
    # torch.save(classification_model, FINETUNED_MODEL_DIR)
    print("Complete!")
if __name__ == "__main__":  
    main()