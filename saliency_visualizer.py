import sys

unwanted_path = '/mmfs1/home/hsethu/.local/lib/python3.9/site-packages'
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)

import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
import os

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

from utils.preprocess import preprocess
from utils.finetune import finetune

from model.ClassificationHead import ClassificationHead
from model.ClassificationLM import ClassificationLM

from datasets import load_dataset, Dataset
import urllib.request

from typing import Tuple, Optional, Callable

### SALIENCY VISUALIZER CLASS ###
class SaliencyVisualizerGeneral():
  """
  Creates a visual representation of the saliency map for an input given a model.
  """
  def __init__(self, model: any, embed_fn: Callable, tokenizer: any, embedder: Optional[any] = None) -> None:
      """
      Initialize a generalized saliency visualizer.

      Args:
          model (any): A sentiment classification model that can accept text and classify it into n labels.
          embed_fn (Callable): A method that can accept a string and return word embeddings for each word.
          tokenizer (any): An object that can tokenize strings in a format acceptable by your embedder/embed function.
          embedder (Optional[any] = None): Embedding model for embed_fn, if needed.
      """
      self.model = model
      self.embed_fn = embed_fn
      self.tokenizer = tokenizer
      self.embedder = embedder

  def viz_saliency_distribution(self, sentence: str, saliency_dist: list[float], label: int, true_label: int, int_to_label: dict, path, model_name, dataset, data_dir) -> None:
    """
    Create a plot that shows the saliency (importance for label prediction) for each word in the input.

    Args:
        sentence (str): Input sentence.
        saliency_dist (list[float]): Categorical probability distribution for saliency of words.
        label (int): Classification of sentence.
        int_to_label (dict): Mapping from integer label to string.
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(1 * len(sentence.split()), 0.5 * len(sentence.split()))
    plt.rc('font', size=14) 
    ax.bar(range(len(saliency_dist)), saliency_dist)
    ax.set_xlabel("Words")
    ax.set_ylabel("Word Saliency")
    ax.set_xticks(range(len(saliency_dist)))
    ax.set_xticklabels(sentence.split(), rotation=-45)
    ax.set_ylim(0,1)
    ax.set_title(f"Model: {model_name}, dataset: {dataset}/{data_dir},\n Label: {int_to_label[label]}, true label: {int_to_label[true_label]}")
    # path = "/gscratch/ark/hari/msc/results/test/img.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    print("saved img")

  def calc_saliency(self, input: str) -> Tuple[list[float], str]:
    """
    Calculate a saliency map for a given input.

    Args:
        input (str): String input for which to calculate the saleincy map w.r.t self.model.
    """
    print("calc")
    # Run model and calculate gradients for word_embeddings
    logits, word_embeddings = self._model_forward(input)
    loss = torch.max(logits)
    loss.backward()

    # Remove start and end tokens
    token_grads = word_embeddings.grad.squeeze()[1:-1]

    # Compute squared L2 norm of token gradients and softmax
    saliency_likelihood = torch.linalg.norm(token_grads, dim=1)**2
    saliency_probs = torch.nn.functional.softmax(saliency_likelihood.detach(), dim=0).cpu().numpy()

    # Group together subwords
    combined_saliency_probs = []
    subword_groups = [self.tokenizer.encode(x, add_special_tokens=False) for x in input.split()]
    for group_len in list(map(len, subword_groups)):
      word_saliency = np.sum(saliency_probs[:group_len])
      saliency_probs = saliency_probs[group_len:]
      combined_saliency_probs.append(word_saliency)
      
    # Return saliency distribution for each word and class
    return (combined_saliency_probs, torch.argmax(logits).item())

  def _model_forward(self, input: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run an input through a model, ensuring that there exist word embeddings that
    are in the PyTorch computation graph for later backprop.

    Args:
        input (str): The sentence to calculate logits and embeddings for.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of model prediction logits and
        word embeddings from the original input that exist on the same PyTorch computational
        graph.
    """
    print("forward")
    # Generate tokens and word embeddings
    word_embeddings = self.embed_fn(input, self.tokenizer, self.embedder)

    # Ensure gradients are retained for the embeddings
    word_embeddings.requires_grad_()
    word_embeddings.retain_grad()

    # Model forward pass
    classification_idx = -1
    outputs = self.model.pretrained_model(inputs_embeds=word_embeddings, output_hidden_states=True)
    model_out = self.model.head(outputs.hidden_states[-1][:, classification_idx, ...])
    print(model_out)
    print(word_embeddings.shape)

    return model_out, word_embeddings
  

### EMBED FUNCTION ###
def embed_function(input: str, tokenizer: any, embedder: Optional[any] = None):
    """
    Turn a sentence into a set of word embeddings

    Args:
        input (str): The sentence to be embedded.
        tokenizer (any): An object that can tokenize strings in a format acceptable by your embedder/embed function.
        embedder (Optional[any] = None): An embedding model, if needed.

    Returns:
        torch.Tensor: A (num_words, embed_dim) vector that has an embedding for each word.
    """
    print("emb")
    encoded_input = tokenizer(input, return_tensors='pt')
    word_embeddings = embedder(encoded_input["input_ids"].cuda())
    return word_embeddings

### MAIN ###
def main():
    # Define huggingface model/task
    # MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    

    TOKEN = "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"
    DATASET_CACHE="/gscratch/ark/hari/generative-classification/generative-classification/models/datasets/"

    dataset_name = "stanfordnlp/snli"
    data_dir = "plain_text"
    # model_name = "meta-llama/Llama-2-7b-hf"
    model_name = "facebook/opt-1.3b"
    path = f"/gscratch/ark/hari/msc/results/{dataset_name}/{data_dir}/{model_name}/"
    model = torch.load(path + "model.pt").cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)

    # Note: Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive
    dataset = load_dataset(dataset_name, data_dir, cache_dir=DATASET_CACHE, split="validation")

    plt.rcParams['font.family'] = 'monospace'
    print("loaded model")
    viz = SaliencyVisualizerGeneral(model, embed_function, tokenizer, model.pretrained_model.get_input_embeddings())

    int_to_label = {
        0: "Entailment",
        1: "Neutral",
        1: "Contradiction"
    }
    sentence = "Two women are embracing while holding to go packages.The sisters are hugging goodbye while holding to go packages after just eating lunch."
    true_label = 0
    saliency_dist, label = viz.calc_saliency(sentence)
    viz.viz_saliency_distribution(sentence, saliency_dist, label, true_label, int_to_label, path+f"{sentence[:10]}-{sentence[-10:]}.png", model_name, dataset_name, data_dir)

if __name__ == "__main__":
    main()