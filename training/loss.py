import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Args:
        temperature (float): scaling factor for similarity scores.
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: torch.Tensor of shape [batch_size, embedding_dim]
            labels: torch.Tensor of shape [batch_size] or [batch_size, 1]

        Returns:
            torch.Tensor: scalar loss value
        """
        if labels.dim() == 2:
            labels = labels.squeeze(1)  # ensure shape [batch_size]

        device = features.device

        # Compute pairwise similarity
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()  # stability trick

        # Exclude self-comparisons
        logits_mask = torch.ones_like(logits).fill_diagonal_(0)

        # Positive mask: pairs with the same label
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device) * logits_mask

        # Log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Mean over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        return -mean_log_prob_pos.mean()

def get_cost_function(objective):

    if objective == "classification":
       cost_function = torch.nn.CrossEntropyLoss()
    if objective == "contrastive":
       cost_function = SupConLoss(temperature=0.07)
    
    return cost_function