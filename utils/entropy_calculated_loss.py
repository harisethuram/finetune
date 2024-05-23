import sys

unwanted_path = '/mmfs1/home/hsethu/.local/lib/python3.9/site-packages'
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

class InverseExpLoss(nn.Module):
    def forward(self, output, target, embeddings, lambda_entropy=0.1, k=10):
        cross_entropy_loss = F.cross_entropy(output, target)

        # Compute gradients with respect to embeddings
        output.retain_grad()
        cross_entropy_loss.backward(retain_graph=True)
        grads = embeddings.grad

        # Calculate L2 norm of gradients
        top_l2_norms = torch.topk(torch.norm(grads, dim=-1), k, dim=-1).values

        # Normalize the gradient scoress
        normalized_scores = (top_l2_norms / torch.sum(top_l2_norms) + 1e-5)

        # Compute entropy
        entropy = -torch.sum(normalized_scores * torch.log(normalized_scores + 1e-8))

        # Combine cross-entropy loss with entropy using inverse exponential to weight smaller entropy values as more costly
        total_loss = cross_entropy_loss + lambda_entropy * torch.exp(-entropy)

        return total_loss


class InverseLoss(nn.Module):
    def forward(self, output, target, embeddings, lambda_entropy=0.1, k=10):
        cross_entropy_loss = F.cross_entropy(output, target)

        # Compute gradients with respect to embeddings
        output.retain_grad()
        cross_entropy_loss.backward(retain_graph=True)
        grads = embeddings.grad

        # Calculate L2 norm of gradients
        top_l2_norms = torch.topk(torch.norm(grads, dim=-1), k, dim=-1).values

        # Normalize the gradient scoress
        normalized_scores = (top_l2_norms / torch.sum(top_l2_norms) + 1e-5)

        # Compute entropy
        entropy = -torch.sum(normalized_scores * torch.log(normalized_scores + 1e-8))

        # Combine cross-entropy loss with entropy using inverse to weight smaller entropy values as more costly
        total_loss = cross_entropy_loss + lambda_entropy / entropy

        return total_loss