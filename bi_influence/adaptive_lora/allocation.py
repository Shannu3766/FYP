import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def allocate_ranks_bi(
    scores: Dict[str, float],
    total_rank: int,
    tau: float = 1.0,
    min_rank: int = 4
) -> Dict[str, int]:
    if not scores:
        return {}

    layer_names = list(scores.keys())
    s = torch.tensor([scores[name] for name in layer_names], dtype=torch.float32)
    s_min, s_max = s.min(), s.max()
    if (s_max - s_min) > 1e-6:
        s = (s - s_min) / (s_max - s_min)
    probs = torch.softmax(s / tau, dim=0)
    raw_ranks = probs * total_rank

    int_ranks = torch.floor(raw_ranks).int()
    remainder = total_rank - int_ranks.sum()
    if remainder > 0:
        residuals = raw_ranks - int_ranks
        _, top_indices = torch.topk(residuals, k=int(remainder.item()))
        int_ranks[top_indices] += 1

    int_ranks = torch.clamp(int_ranks, min=min_rank)

    if int_ranks.sum() > total_rank + len(scores) * (min_rank - 1):
        logger.warning(f"Total allocated rank ({int_ranks.sum()}) exceeds budget ({total_rank}).")

    return {name: rank.item() for name, rank in zip(layer_names, int_ranks)}
