from typing import Tuple

import torch
import torch.nn.functional as F


def ranking_loss(
    logits: torch.Tensor, target_idx: int, margin: float, lambda_margin: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the combined loss, cross entropy, and hard-negative margin loss."""
    target = torch.tensor([target_idx], dtype=torch.long, device=logits.device)
    ce_loss = F.cross_entropy(logits.unsqueeze(0), target)
    if logits.numel() <= 1:
        margin_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
    else:
        pos_score = logits[target_idx]
        neg_mask = torch.ones_like(logits, dtype=torch.bool)
        neg_mask[target_idx] = False
        hard_neg = logits[neg_mask].max()
        margin_loss = F.relu(margin - pos_score + hard_neg)
    total_loss = ce_loss + lambda_margin * margin_loss
    return total_loss, ce_loss, margin_loss
