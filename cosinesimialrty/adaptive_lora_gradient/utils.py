import os
import csv
from typing import Dict, Any
import torch
from peft.tuners.lora import LoraLayer
import logging

logger = logging.getLogger(__name__)

def get_lora_layers(model: torch.nn.Module) -> Dict[str, LoraLayer]:
    """
    Finds all modules in the model that are instances of peft.tuners.lora.LoraLayer.

    Args:
        model: The PEFT model.

    Returns:
        A dictionary mapping qualified layer names to the LoraLayer module.
    """
    # Note: We look for LoraLayer, which is the base class for peft.lora.Linear,
    # peft.lora.Embedding, peft.lora.Conv2d, etc.
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoraLayer)
    }

def save_epoch_log(
    log_file: str, 
    epoch: int, 
    ranks: Dict[str, int], 
    scores: Dict[str, float]
):
    """
    Appends the rank allocation results for the current epoch to a CSV log file.
    
    Args:
        log_file: Path to the CSV file.
        epoch: The current epoch number.
        ranks: Dictionary of {layer_name: allocated_rank}.
        scores: Dictionary of {layer_name: importance_score}.
    """
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    fieldnames = ['epoch', 'layer_name', 'importance_score', 'allocated_rank']
    
    # Check if file exists to write header
    file_exists = os.path.isfile(log_file)
    
    try:
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                
            for layer_name in ranks.keys():
                writer.writerow({
                    'epoch': epoch,
                    'layer_name': layer_name,
                    'importance_score': scores.get(layer_name, 0.0),
                    'allocated_rank': ranks.get(layer_name, 0)
                })
    except IOError as e:
        logger.error(f"Failed to write to log file {log_file}: {e}")