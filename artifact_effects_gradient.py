import sys

unwanted_path = '/mmfs1/home/hsethu/.local/lib/python3.9/site-packages'
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset

import csv
import urllib.request
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from typing import Tuple, Optional, Callable
import sys
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.preprocess import preprocess
from keys import HF_TOKEN

import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--finetuned_model_dir")
parser.add_argument("--model_name", help="to load in the corresponding tokenizer")
parser.add_argument("--results_dir")
parser.add_argument("--dataset")
parser.add_argument("--data_dir")
parser.add_argument("--dataset_cache")
parser.add_argument("--split", default="validation")
parser.add_argument("--cols")
parser.add_argument("--null_lab", type=int, default=-1)
parser.add_argument("--csv_file")
args = parser.parse_args()
print(args)

# NEED TO HAVE .csv OF SST2 WORDS

def masked_saliency_calc(model, tokenizer, sentences, input_ids, attention_masks, labels, classification_idx: int, result_dir, dataset_length, csv_file, title, k=5, copy_b_size=16, device="cuda"):
    ### Varun's Code ###
    # Get .csv of ALL words
    all_df = pd.read_csv(csv_file)
    # print(labels)
    # print(labels.shape, input_ids.shape)

    # Filter into an artifact dataframe
    artifact_df = all_df[all_df["isArtifact"] == True][["Unnamed: 0", "n"]].rename(columns={"Unnamed: 0": "word"})
    all_df = all_df[["Unnamed: 0", "n", "isArtifact"]].rename(columns={"Unnamed: 0": "word"})

    ## ! MODIFY IF NEEDED ! ##
    remove_words = ["the", "a", "and", "of", "s", "that", "in", "with", "an"]
    artifact_df = artifact_df[artifact_df["word"].isin(remove_words) == False]
    all_df = all_df[all_df["word"].isin(remove_words) == False]
    ## ! MODIFY IF NEEDED ! ##
    # print(output)
    print("All Words")
    print(all_df.head())
    print(len(all_df))
    print("Just artifacts")
    print(artifact_df.head())
    print(len(artifact_df))

    # Unconditioned probability of word being an artifact
    p_artifact = artifact_df["n"].sum() / all_df["n"].sum()
    p_non_artifact = 1 - p_artifact

    print(f"P(Artifact) = {p_artifact}; P(Not Artifact) = {p_non_artifact}")

    num_topk_words = 0
    num_artifacts_in_topk = 0
    num_non_artifacts_in_topk = 0
    diff_sum = []
    ### End Varun's Code ###

    for i in tqdm(range(dataset_length)):
        sentence = sentences[i]
        input_id = torch.unsqueeze(input_ids[i], dim=0)
        attention_mask = torch.unsqueeze(attention_masks[i], dim=0)
        label = torch.unsqueeze(labels[i], dim=0)
      
        ### Hari's Code ###

        seq_len = torch.sum(attention_mask).item()

        # compute the entropy
        output, embs = model(input_id, attention_mask, classification_idx=classification_idx, want_embs=True)
        prob = F.softmax(output, dim=-1)[..., label]
        loss = -torch.log(prob)

        output.retain_grad()
        loss.backward()
        grads = embs.grad
        # diffs = torch.absolute(prob)# F.softmax(torch.topk(torch.absolute(prob - probs_copy), k).values)
        diffs = torch.norm(grads, dim=-1)[...,:seq_len]

        # Normalize the gradient scoress
        # prob = F.softmax(output, dim=-1)[..., label]
        # diffs = torch.absolute(prob - probs_copy)# F.softmax(torch.topk(torch.absolute(prob - probs_copy), k).values)

        combined_saliency_probs = []
        subword_groups = [tokenizer.encode(x, add_special_tokens=False) for x in sentence.strip().split()]
        for group_len in list(map(len, subword_groups)):
            word_saliency = torch.sum(diffs[:group_len])
            diffs = diffs[group_len:]
            combined_saliency_probs.append(word_saliency) #This should be word saliency, not token saliency
        ### End Hari's Code ###

        ### Varun's Code ###
        words = sentence.split()
        topk_words = []
        # print(torch.Tensor(combined_saliency_probs).shape)
        for tk in torch.flip(torch.argsort(torch.Tensor(combined_saliency_probs)), dims=(0,))[:k]:
            topk_words.append(words[tk])

        # Get total number of artifacts in whole sentence
        num_artifacts = len(set(words).intersection(set(artifact_df["word"])))

        # Get words + counts dataframes for the topk, and topk and artifact
        topk_all_words = all_df[all_df["word"].isin(topk_words) == True]
        topk_artifact_words = topk_all_words[topk_all_words["isArtifact"] == True]

        # P(word in topk)
        p_topk = len(topk_words) / len(words)
        # P(word in topk | word is artifact)
        p_topk_given_artifact = (len(topk_artifact_words) / num_artifacts) if num_artifacts > 0 else 0
        # Difference in probabilities
        diff_sum.append(p_topk_given_artifact - p_topk)

        # Count number of words seen
        num_topk_words += len(topk_words)

        # Increase appropriate counter
        num_artifacts_in_topk += len(topk_artifact_words)
        num_non_artifacts_in_topk += len(topk_words)-len(topk_artifact_words)

    # Sentence-level
    diff_sum = torch.Tensor(diff_sum)
    with open(result_dir+"gradient_artifacts.txt", "w+") as f:
        f.write(f"Difference when conditioned on artifacts: {torch.mean(torch.Tensor(diff_sum))}\n")

        # Dataset-level
        f.write(f"Expected number of artifacts in topk under null distribution: {num_topk_words * p_artifact}\n")
        f.write(f"Actual Number of artifacts in topk: {num_artifacts_in_topk}")

        f.write(f"Expected number of non-artifacts in topk under null distribution: {num_topk_words * p_non_artifact}\n")
        f.write(f"Actual number of non-artifacts in topk: {num_non_artifacts_in_topk}\n")

        

        # Final Results
        labels = ["Expected Artifacts", "Actual Artifacts", "Expected Non-Artifacts", "Actual Non-Artifacts"]
        counts = [num_topk_words * p_artifact, num_artifacts_in_topk, num_topk_words * p_non_artifact, num_non_artifacts_in_topk]
        bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:blue']
        
        f.write(f"Difference when conditioned on artifacts: {torch.mean(diff_sum)}\n")
        for label, count in zip(labels, counts):
            f.write(f"{label}: {count}\n")
    plt.title(title)
    plt.bar(labels, counts, color=bar_colors)
    plt.xticks(rotation=-45)
    plt.tight_layout()
    plt.savefig(result_dir+"gradient_artifacts.png")
    # plt.show()
    ### End Varun's Code ###
  
def main():
    print("OK")
    if not os.path.exists(args.finetuned_model_dir):
        print("model does not exist!")
        return None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dataset = load_dataset(args.dataset, args.data_dir, split=args.split, cache_dir=args.dataset_cache)
    cols = args.cols.split(",")

    
    model = torch.load(args.finetuned_model_dir).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=HF_TOKEN)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    

    tokenized_dataset = preprocess(raw_dataset, tokenizer, cols, args.null_lab, DEVICE)
    n = len(tokenized_dataset)
    length = len(tokenized_dataset[0][0])
    input_ids = torch.zeros(n, length).int().to(DEVICE)
    attention_masks = torch.zeros(n, length).int().to(DEVICE)
    labels = torch.zeros(n).int().to(DEVICE)

    for i in range(n):
        input_ids[i], attention_masks[i], labels[i] = tokenized_dataset[i]

    # print("[ass]", input_ids.shape)
    sentences = [tokenizer.decode(input_ids[i][input_ids[i] != tokenizer.pad_token_id].int()) for i in range(n)]
    # print(sentences[:10])
    info = args.results_dir.split("/")
    title = f"Expected vs. Actual counts for Artifacts/Non-Artifacts\nin the Top-{5} Most Salient Words\nOn {args.dataset}/{args.data_dir} using {args.model_name}\nW/ {info[-4]}, {info[-3]}, {info[-2]}"
    masked_saliency_calc(model, tokenizer, sentences, input_ids, attention_masks, labels, -1, args.results_dir, n, args.csv_file, title, device=DEVICE)


if __name__ == "__main__":
    main()
    

