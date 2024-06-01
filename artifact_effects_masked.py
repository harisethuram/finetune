import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F

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

# NEED TO HAVE .csv OF SST2 WORDS

def masked_saliency_calc(model, tokenizer, sentences: list, input_ids: list, attention_masks: list, labels: int, classification_idxs: int, output="output.png", k=10, copy_b_size=16, device="cuda"):
    ### Varun's Code ###
    # Get .csv of ALL words
    all_df = pd.read_csv("sst2_all.csv")

    # Filter into an artifact dataframe
    artifact_df = all_df[all_df["isArtifact"] == True][["Unnamed: 0", "n"]].rename(columns={"Unnamed: 0": "word"})
    all_df = all_df[["Unnamed: 0", "n", "isArtifact"]].rename(columns={"Unnamed: 0": "word"})

    ## ! MODIFY IF NEEDED ! ##
    remove_words = [] #["the", "a", "and", "of", "s", "that", "in", "with", "an"]
    artifact_df = artifact_df[artifact_df["word"].isin(remove_words) == False]
    all_df = all_df[all_df["word"].isin(remove_words) == False]
    ## ! MODIFY IF NEEDED ! ##

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

    for sentence, input_id, attention_mask, classification_idx, label in zip(sentences, input_ids, attention_masks, classification_idxs, labels):
      
      ### Hari's Code ###
      seq_len = torch.sum(attention_mask)
      copy_masks = attention_mask.repeat((seq_len, 1))
      input_ids_copy = input_ids.repeat((copy_b_size, 1))
      copy_masks[torch.arange(seq_len), torch.arange(seq_len)] = 0
      probs_copy = torch.zeros(seq_len).to(device) # probability of correct class removing the ith token

      for i in range(int(seq_len/copy_b_size)): # get the probabilities if you mask out the ith word
          masks_i = copy_masks[i * copy_b_size : min((i+1) * copy_b_size, seq_len)]
          logits_i = model(input_ids_copy[:masks_i.shape[0], :], masks_i, classification_idx=classification_idx)
          probs_copy[i * copy_b_size : min((i+1) * copy_b_size, seq_len)] = torch.squeeze(F.softmax(logits_i, dim=-1)[..., label])

      # compute the entropy
      output = model(input_id, attention_mask, classification_idx=classification_idx)
      prob = F.softmax(output, dim=-1)[..., label]
      diffs = torch.absolute(prob - probs_copy)# F.softmax(torch.topk(torch.absolute(prob - probs_copy), k).values)

      combined_saliency_probs = []
      subword_groups = [tokenizer.encode(x, add_special_tokens=False) for x in sentence.trim().split()]
      for group_len in list(map(len, subword_groups)):
        word_saliency = np.sum(diffs[:group_len])
        diffs = diffs[group_len:]
        combined_saliency_probs.append(word_saliency) #This should be word saliency, not token saliency
      ### End Hari's Code ###

      ### Varun's Code ###
      words = sentence.split()
      topk_words = []
      for tk in np.argsort(np.array(combined_saliency_probs))[::-1][:k]:
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
    print(f"Difference when conditioned on artifacts: {np.mean(diff_sum)}")

    # Dataset-level
    print(f"Expected number of artifacts in topk under null distribution: {num_topk_words * p_artifact}")
    print(f"Actual Number of artifacts in topk: {num_artifacts_in_topk}")

    print(f"Expected number of non-artifacts in topk under null distribution: {num_topk_words * p_non_artifact}")
    print(f"Actual number of non-artifacts in topk: {num_non_artifacts_in_topk}")

    # Final Results
    labels = ["Expected Artifacts", "Actual Artifacts", "Expected Non-Artifacts", "Actual Non-Artifacts"]
    counts = [num_topk_words * p_artifact, num_artifacts_in_topk, num_topk_words * p_non_artifact, num_non_artifacts_in_topk]
    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:blue']
    plt.bar(labels, counts, color=bar_colors)
    plt.xticks(rotation=-45)
    plt.title(f"Expected vs. Actual counts for Artifacts and Non-Artifacts in the Top-{k} Most Salient Words")
    print(f"Difference when conditioned on artifacts: {np.mean(diff_sum)}")

    plt.savefig(output)
    plt.show()
    ### End Varun's Code ###
  
