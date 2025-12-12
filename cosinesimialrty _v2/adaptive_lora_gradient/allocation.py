import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def _largest_remainder_rounding(
    raw_ranks: torch.Tensor, 
    budget: int
) -> torch.Tensor:
    """
    Rounds raw rank allocations to integers while perfectly preserving the sum.
    This implements "rounding top residuals" (Algorithm 2, line 14). 
    
    Args:
        raw_ranks: A tensor of floating-point rank allocations.
        budget: The exact integer total sum that must be preserved.

    Returns:
        A tensor of integer ranks.
    """
    # This is r_i = floor(R * alpha_i) (Algorithm 2, line 13)
    int_ranks = torch.floor(raw_ranks).int()
    
    # This is the adjustment step (Algorithm 2, line 14)
    remainder = budget - int_ranks.sum()
    
    if remainder < 0:
        # This can happen due to floating point inaccuracies
        logger.warning(f"Rounding remainder is negative ({remainder}). Adjusting...")
        _, top_indices = torch.topk(raw_ranks, k=int(torch.abs(remainder).item()))
        int_ranks[top_indices] -= 1
        return int_ranks
        
    elif remainder > 0:
        # Get the "residuals" (the fractional parts)
        residuals = raw_ranks - int_ranks
        
        # "rounding top residuals"
        _, top_indices = torch.topk(residuals, k=int(remainder.item()))
        int_ranks[top_indices] += 1
            
    return int_ranks
def allocate_ranks_bi(
    scores: Dict[str, float],
    total_rank: int,
    tau: float = 0.3,        # make distribution sharper
    eps: float = 1e-8,
    min_rank:int=1,
) -> Dict[str, int]:
    """
    Allocates ranks to layers based on their BI importance scores.
    Includes normalization + temperature sharpening.
    """
    if not scores:
        return {}

    layer_names = list(scores.keys())
    s = torch.tensor([scores[name] for name in layer_names], dtype=torch.float32)

    # --- 1️⃣ Normalize scores to [0, 1]
    s_min, s_max = s.min(), s.max()

    if (s_max - s_min) < eps:
        # all scores equal → give uniform rank
        uniform_rank = max(1, total_rank // len(scores))
        return {name: uniform_rank for name in layer_names}

    s = (s - s_min) / (s_max - s_min) 

    s_temp = s / tau
    probs = torch.softmax(s_temp, dim=0)

    # --- 3️⃣ Allocate and round ranks
    raw_ranks = probs * total_rank
    int_ranks = torch.floor(raw_ranks).int()

    remainder = total_rank - int_ranks.sum()
    if remainder > 0:
        residuals = raw_ranks - int_ranks
        _, top_indices = torch.topk(residuals, k=int(remainder.item()))
        int_ranks[top_indices] += 1

    # --- 4️⃣ Ensure no layer gets zero rank
    int_ranks = torch.clamp(int_ranks, min=1)

    return {name: rank.item() for name, rank in zip(layer_names, int_ranks)}
