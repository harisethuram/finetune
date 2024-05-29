import sys

unwanted_path = '/mmfs1/home/hsethu/.local/lib/python3.9/site-packages'
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

# compute the entropy
def entropy_top_k(scores, k):
    top_l2_norms = torch.topk(scores, k, dim=-1).values
    normalized_scores = F.softmax(top_l2_norms, dim=-1)

    return -torch.sum(normalized_scores * torch.log(normalized_scores + 1e-8))

# a bunch of ways to combine entropy and cross entropy loss, penalizing a low entropy
def inverse_entropy(cross_entropy_loss, entropy, lambda_entropy=0.1):
    return cross_entropy_loss / (lambda_entropy * entropy + 1e-8)

def exp_subtract_entropy(cross_entropy_loss, entropy, lambda_entropy=0.1):
    return torch.exp(cross_entropy_loss - lambda_entropy * entropy)

def subtract_entropy(cross_entropy_loss, entropy, lambda_entropy=0.1):
    return cross_entropy_loss - lambda_entropy * entropy

# the class
class EntropyFunc(nn.Module):
    def __init__(self, entropy_func, lambda_entropy=0.1):
        super(EntropyFunc, self).__init__()
        entropy_func_dict = {"inverse_entropy": inverse_entropy, 
                             "subtract_entropy": subtract_entropy, 
                             "exp_subtract_entropy": exp_subtract_entropy}
        self.entropy_func = entropy_func_dict[entropy_func]
        self.lambda_entropy = lambda_entropy

    def forward(self, CE_loss, entropy):
        return self.entropy_func(CE_loss, entropy, self.lambda_entropy)
