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
    """
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
    """
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

def resize_lora_layer_svd(
    layer: LoraLayer, 
    new_rank: int, 
    lora_alpha: int, 
    adapter_name: str = "default",
    **kwargs
):
    """
    Resizes a LoRA layer using SVD to preserve the learned weights.
    """
    
    # 1. Capture current weights and scaling
    with torch.no_grad():
        if adapter_name not in layer.lora_A:
            return
            
        old_r = layer.r[adapter_name]
        old_alpha = layer.lora_alpha[adapter_name]
        old_scaling = old_alpha / old_r
        
        # Get weights
        A_old = layer.lora_A[adapter_name].weight
        B_old = layer.lora_B[adapter_name].weight
        
        # Compute effective weight
        W_delta = (B_old @ A_old) * old_scaling
        
        # 2. Perform SVD
        dtype = A_old.dtype
        U, S, Vh = torch.linalg.svd(W_delta.float(), full_matrices=False)
        
        # 3. Truncate to new rank
        k = new_rank
        k = min(k, S.size(0))
        
        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]
        
        # 4. Calculate new A and B
        sqrt_S = torch.diag(torch.sqrt(S_k))
        B_new = (U_k @ sqrt_S).to(dtype)
        A_new = (sqrt_S @ Vh_k).to(dtype)
        
        # 5. Adjust for NEW scaling
        new_scaling = lora_alpha / new_rank
        scale_correction = 1.0 / (new_scaling ** 0.5)
        
        B_new *= scale_correction
        A_new *= scale_correction
        

    if 'init_lora_weights' in kwargs:
        kwargs.pop('init_lora_weights')

    layer.update_layer(
        adapter_name=adapter_name,
        r=new_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True, 
        **kwargs
    )
    
    with torch.no_grad():
        if k < new_rank:
             layer.lora_A[adapter_name].weight.data.zero_()
             layer.lora_B[adapter_name].weight.data.zero_()
             layer.lora_A[adapter_name].weight.data[:k, :] = A_new
             layer.lora_B[adapter_name].weight.data[:, :k] = B_new
        else:
             layer.lora_A[adapter_name].weight.data = A_new
             layer.lora_B[adapter_name].weight.data = B_new