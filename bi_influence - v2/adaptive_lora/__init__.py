"""
adaptive_lora: A package for per-epoch adaptive LoRA rank allocation 
based on Block Influence (BI) scores, compatible with Hugging Face.
"""

__version__ = "0.1.0"

from .callbacks import AdaptiveLoRACallback
from .importance import compute_bi_scores
from .allocation import allocate_ranks_bi

__all__ = [
    "AdaptiveLoRACallback",
    "compute_bi_scores",
    "allocate_ranks_bi"
]