import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging
from tqdm.auto import tqdm
from .utils import get_lora_layers

logger = logging.getLogger(__name__)

def compute_gradient_importance_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: Optional[int] = None, 
    batch_size: int = 8                
) -> Dict[str, float]:

    model.eval()
    lora_layers = get_lora_layers(model)
    if not lora_layers:
        logger.warning("No LoRA layers found. Returning empty scores.")
        return {}

    if batch_size != dataloader.batch_size:
        logger.info(f"Adjusting DataLoader batch size from {dataloader.batch_size} to {batch_size}")
        dataloader = DataLoader(
            dataset=dataloader.dataset,
            batch_size=batch_size,
            shuffle=False, 
            collate_fn=dataloader.collate_fn, 
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory
        )

    accumulated_scores = {name: 0.0 for name in lora_layers.keys()}
    
    current_batch_activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                out.retain_grad() 
                current_batch_activations[name] = out
            elif isinstance(out, (tuple, list)):
                for elem in out:
                    if isinstance(elem, torch.Tensor):
                        elem.retain_grad()
                        current_batch_activations[name] = elem
                        break
        return hook

    hooks = []
    for name, layer in lora_layers.items():
        try:
            hooks.append(layer.register_forward_hook(make_hook(name)))
        except Exception as e:
            logger.warning(f"Failed to register hook for {name}: {e}")

    total_steps = len(dataloader)
    if num_batches is not None:
        total_steps = min(num_batches, total_steps)

    batches_processed = 0

    model.zero_grad(set_to_none=True)
    
    with torch.enable_grad():
        for step, batch in tqdm(enumerate(dataloader), total=total_steps, desc="Computing Importance", leave=False):
            
            if num_batches is not None and step >= num_batches:
                break

            if hasattr(batch, "items"):
                batch_input = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(**batch_input)
            elif isinstance(batch, (list, tuple)):
                batch_input = [b.to(device) for b in batch if isinstance(b, torch.Tensor)]
                outputs = model(*batch_input)
            else: 
                outputs = model(batch.to(device))

            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            elif hasattr(outputs, "loss"):
                loss = outputs.loss
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                loss = outputs[0]
            else:
                continue

            loss.backward()
            for name, act in current_batch_activations.items():
                if act.grad is None:
                    continue

                a_flat = act.detach().view(-1, act.size(-1))
                g_flat = act.grad.detach().view(-1, act.size(-1))
                importance = (a_flat * g_flat).sum(dim=1).abs().mean().item()
                

                accumulated_scores[name] += importance

            current_batch_activations.clear()
            model.zero_grad(set_to_none=True)
            batches_processed += 1
            

    for h in hooks:
        h.remove()

    if batches_processed == 0:
        return {k: 0.0 for k in lora_layers.keys()}


    final_scores = {k: v / batches_processed for k, v in accumulated_scores.items()}

    s_vals = torch.tensor(list(final_scores.values()), dtype=torch.float32)
    eps = 1e-8
    s_min, s_max = s_vals.min(), s_vals.max()

    if (s_max - s_min) < eps:
        normed = {k: 0.0 for k in final_scores.keys()}
    else:
        s_norm = (s_vals - s_min) / (s_max - s_min)
        normed = {k: float(v) for k, v in zip(final_scores.keys(), s_norm)}

    model.train()
    return normed