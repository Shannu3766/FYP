import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, Union
import logging
from adaptive_lora.utils import get_lora_layers

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR) 

def compute_bi_scores(
    model: torch.nn.Module,
    val_data: Union[DataLoader, torch.utils.data.Dataset],
    device: torch.device,
    collate_fn=None,
    batch_size: int = 4,
) -> Dict[str, float]:

    model.eval()

    if isinstance(val_data, DataLoader):
        dataloader = val_data
    else:
        dataloader = DataLoader(
            val_data,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )

    lora_layers = get_lora_layers(model)
    if not lora_layers:
        logger.warning("No LoRA layers found in the model. Returning empty scores.")
        return {}

    running_stats = {name: {"sum": 0.0, "count": 0} for name in lora_layers}
    hooks = []

    def hook_factory(name):
        def hook(module, input_act, output_act):
            try:
                if input_act is None or output_act is None:
                    return
                x_in = input_act[0].detach().to(torch.float32)
                x_out = output_act.detach().to(torch.float32)

                in_flat = x_in.view(-1, x_in.size(-1))
                out_flat = x_out.view(-1, x_out.size(-1))

                d = min(in_flat.size(1), out_flat.size(1))
                in_flat = in_flat[:, :d]
                out_flat = out_flat[:, :d]

                cos_sim = F.cosine_similarity(in_flat, out_flat, dim=1)
                cos_sim = torch.nan_to_num(cos_sim, nan=0.0, posinf=0.0, neginf=0.0)
                cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                mean_cos = max(min(cos_sim.mean().item(), 1.0), -1.0)

                running_stats[name]["sum"] += mean_cos
                running_stats[name]["count"] += 1
            except Exception:
                pass 
        return hook

    for name, layer in lora_layers.items():
        try:
            hooks.append(layer.register_forward_hook(hook_factory(name)))
        except Exception:
            pass

    logger.info(f"Starting BI forward pass (batch_size={batch_size})...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing BI scores", leave=True):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model(**batch)

    for h in hooks:
        h.remove()

    bi_scores = {}
    for name in lora_layers:
        stats = running_stats[name]
        if stats["count"] == 0:
            bi_scores[name] = 0.0
            continue
        rho_i = stats["sum"] / stats["count"]
        rho_i = max(min(rho_i, 1.0), -1.0)
        s_i = 1.0 - rho_i
        s_i = max(0.0, min(s_i, 1.0))  
        bi_scores[name] = s_i

    model.train()
    logger.info("✅ BI computation complete.")
    return bi_scores
