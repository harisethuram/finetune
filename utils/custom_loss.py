import sys

unwanted_path = '/mmfs1/home/hsethu/.local/lib/python3.9/site-packages'
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)
    
import torch
import torch.nn as nn

class CompetentLoss(nn.Module):
    def __init__(self, ):
        super(CompetentLoss, self).__init__()

    def forward(self, output, target, embeddings, lambda_entropy=0.1):
        cross_entropy_loss = F.cross_entropy(output, target)
        
        # Compute gradients with respect to embeddings
        output.retain_grad()
        cross_entropy_loss.backward(retain_graph=True)
        grads = embeddings.grad
        
        # Calculate L2 norm of gradients
        l2_norms = torch.norm(grads, dim=-1)
        
        # Normalize the gradient scores
        normalized_scores = l2_norms / torch.sum(l2_norms)
        
        # Compute entropy
        entropy = -torch.sum(normalized_scores * torch.log(normalized_scores + 1e-8))
        
        # Combine cross-entropy loss with entropy
        total_loss = cross_entropy_loss + lambda_entropy * entropy
        
        return total_loss