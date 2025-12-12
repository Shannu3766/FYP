import os
import csv
from typing import Dict
import torch
from peft.tuners.lora import LoraLayer
import logging

logger = logging.getLogger(__name__)

def get_lora_layers(model: torch.nn.Module) -> Dict[str, LoraLayer]:
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
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    fieldnames = ['epoch', 'layer_name', 'importance_score', 'allocated_rank']    
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