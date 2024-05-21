import sys

unwanted_path = '/mmfs1/home/hsethu/.local/lib/python3.9/site-packages'
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, output, target, embeddings, lambda_entropy=0.1, k=10):
        return F.cross_entropy(output, target)
        
class CompetentLoss(nn.Module):
    def __init__(self, ):
        super(CompetentLoss, self).__init__()

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

        # Combine cross-entropy loss with entropy
        total_loss = cross_entropy_loss + lambda_entropy * entropy
        
        return total_loss