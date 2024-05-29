import sys

unwanted_path = '/mmfs1/home/hsethu/.local/lib/python3.9/site-packages'
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy_loss(output, target):
    loss = F.cross_entropy(output, target)
    output.retain_grad()
    loss.backward(retain_graph=True)
    return loss

def entropy(grads, k):
    top_l2_norms = torch.topk(torch.norm(grads, dim=-1), k, dim=-1).values
    normalized_scores = (top_l2_norms / torch.sum(top_l2_norms) + 1e-5)

    return -torch.sum(normalized_scores * torch.log(normalized_scores + 1e-8))

def inverse_exp_entropy(grads, k):
    return torch.exp(-entropy(grads, k))

def inverse_entropy(grads, k):
    return 1 / entropy(grads, k)

class HelperCustomLoss(nn.Module):
    def __init__(self, loss_func, reg_func):
        super(HelperCustomLoss, self).__init__()
        self.loss_func = loss_func
        self.reg_func = reg_func

    def forward(self, output, target, embeddings, lambda_entropy=0.1, k=10):
        loss = self.loss_func(output, target)
        grads = embeddings.grad
        return loss + lambda_entropy * self.reg_func(grads, k)

class InverseExpLoss(HelperCustomLoss):
    def __init__(self):
        super(InverseExpLoss, self).__init__(cross_entropy_loss, inverse_exp_entropy)

class InverseLoss(HelperCustomLoss):
    def __init__(self):
        super(InverseLoss, self).__init__(cross_entropy_loss, inverse_entropy)
