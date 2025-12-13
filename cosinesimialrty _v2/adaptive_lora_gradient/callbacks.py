import os
import logging
import torch
from peft.tuners.lora import LoraLayer
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from .importance import compute_gradient_importance_scores
from .allocation import allocate_ranks_bi
from .utils import get_lora_layers, save_epoch_log

logger = logging.getLogger(__name__)

# ============================================================
# 🔧 HELPER: SVD-Based Layer Resizing
# ============================================================
def resize_lora_layer_svd(
    layer: LoraLayer, 
    new_rank: int, 
    lora_alpha: int, 
    adapter_name: str = "default",
    **kwargs
):
    """
    Resizes a LoRA layer using SVD to preserve learned weights.
    """
    with torch.no_grad():
        if adapter_name not in layer.lora_A:
            return
            
        old_r = layer.r[adapter_name]
        
        # Edge case: if rank is 0 or uninitialized
        if old_r == 0: 
             layer.update_layer(adapter_name, new_rank, lora_alpha=lora_alpha, init_lora_weights=True, **kwargs)
             return

        old_alpha = layer.lora_alpha[adapter_name]
        old_scaling = old_alpha / old_r
        
        # Get current weights
        A_old = layer.lora_A[adapter_name].weight
        B_old = layer.lora_B[adapter_name].weight
        
        # Compute effective weight: W = B @ A * scaling
        W_delta = (B_old @ A_old) * old_scaling
        
        # SVD
        dtype = A_old.dtype
        U, S, Vh = torch.linalg.svd(W_delta.float(), full_matrices=False)
        
        # Truncate to new rank
        k = min(new_rank, S.size(0))
        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]
        
        # Reconstruct
        sqrt_S = torch.diag(torch.sqrt(S_k))
        B_new = (U_k @ sqrt_S).to(dtype)
        A_new = (sqrt_S @ Vh_k).to(dtype)
        
        # Scale Correction
        if new_rank > 0:
            new_scaling = lora_alpha / new_rank
            scale_correction = 1.0 / (new_scaling ** 0.5)
            B_new *= scale_correction
            A_new *= scale_correction

    # Remove init arg if present to avoid conflict
    if 'init_lora_weights' in kwargs:
        kwargs.pop('init_lora_weights')

    # Update Structure
    layer.update_layer(
        adapter_name=adapter_name,
        r=new_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True, # Init enabled, we overwrite below
        **kwargs
    )
    
    # Write Weights Back
    with torch.no_grad():
        device = layer.lora_A[adapter_name].weight.device
        if k < new_rank:
             # Growth: Pad with zeros
             layer.lora_A[adapter_name].weight.data.zero_()
             layer.lora_B[adapter_name].weight.data.zero_()
             layer.lora_A[adapter_name].weight.data[:k, :] = A_new.to(device)
             layer.lora_B[adapter_name].weight.data[:, :k] = B_new.to(device)
        else:
             # Compression: Copy truncated
             layer.lora_A[adapter_name].weight.data = A_new.to(device)
             layer.lora_B[adapter_name].weight.data = B_new.to(device)


# ============================================================
# 🧠 Main Callback Class
# ============================================================
class AdaptiveLoRACallback(TrainerCallback):
    def __init__(
        self,
        total_rank: int,
        val_dataloader,
        tau: float = 1.0,
        log_path: str = "./logs",
        verbose: bool = True,
        lora_alpha: int = 16, 
        validate_batch_size: int = 8,
        min_rank: int = 4,
        score_smoothing_beta: float = 0.8, 
        update_interval: int = 1,
        warmup_epochs: int = 0,
        cooldown_epochs: int = 0
    ):
        self.total_rank = total_rank
        self.val_dataloader = val_dataloader
        self.tau = tau
        self.verbose = verbose
        self.log_file = os.path.join(log_path, "adaptive_lora_epoch_logs.csv")
        self.lora_alpha = lora_alpha
        self.validate_batch_size = validate_batch_size
        self.min_rank = min_rank
        
        # Scheduling & Smoothing
        self.score_smoothing_beta = score_smoothing_beta
        self.update_interval = update_interval
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs

        os.makedirs(log_path, exist_ok=True)

        self.latest_scores = {}
        self.ema_scores = {}
        self.latest_ranks = {}

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs
    ):
        epoch = int(state.epoch) + 1 if state.epoch is not None else 1

        if self.verbose:
            print(f"\n--- AdaptiveLoRA: Preparing ranks for Epoch {epoch} ---")

        # --- 1. Scheduling Logic ---
        total_epochs = args.num_train_epochs
        
        if epoch <= self.warmup_epochs:
            if self.verbose: print(f"⏳ Warmup Period ({epoch}/{self.warmup_epochs}). Skipping.")
            return

        if epoch > (total_epochs - self.cooldown_epochs):
            if self.verbose: print(f"❄️ Cooldown Period. Skipping.")
            return

        if (epoch - self.warmup_epochs - 1) % self.update_interval != 0:
            if self.verbose: print(f"⏩ Skipping update (Interval={self.update_interval}).")
            return

        # --- 2. Compute Scores ---
        device = next(model.parameters()).device
        if self.verbose: print("Computing Gradient Importance scores...")
        
        current_scores = compute_gradient_importance_scores(
            model, 
            self.val_dataloader, 
            device, 
            batch_size=self.validate_batch_size
        )
        
        if not current_scores:
            if self.verbose: print("⚠️ No scores found. Skipping.")
            return

        # --- 3. EMA Smoothing ---
        if self.score_smoothing_beta > 0.0:
            if not self.ema_scores:
                self.ema_scores = current_scores
            else:
                for name, score in current_scores.items():
                    prev = self.ema_scores.get(name, score)
                    self.ema_scores[name] = (self.score_smoothing_beta * prev) + \
                                            ((1 - self.score_smoothing_beta) * score)
            scores_to_use = self.ema_scores
            if self.verbose: print(f"📊 Applied Score Smoothing (beta={self.score_smoothing_beta})")
        else:
            scores_to_use = current_scores

        self.latest_scores = scores_to_use

        # --- 4. Allocate Ranks ---
        new_ranks = allocate_ranks_bi(
            scores_to_use, 
            self.total_rank, 
            self.tau, 
            min_rank=self.min_rank
        )

        # --- 5. Apply Updates (SVD) ---
        lora_layers = get_lora_layers(model)
        config = model.peft_config.get("default")
        
        update_kwargs = {
            "use_rslora": getattr(config, "use_rslora", False),
            "use_dora": getattr(config, "use_dora", False),
            "use_qalora": getattr(config, "use_qalora", False),
            "lora_bias": getattr(config, "bias", "none"),
            "qalora_group_size": getattr(config, "qalora_group_size", 64),
        }

        if self.verbose: print(f"Rank Updates (Epoch {epoch}):")

        for name, layer in lora_layers.items():
            new_rank = new_ranks.get(name)
            if new_rank is None: continue

            current_rank = layer.r.get("default", 0)
            score = scores_to_use.get(name, 0.0)

            if current_rank != new_rank:
                if self.verbose:
                    print(f"  - {name}: r={current_rank} → {new_rank} (Score: {score:.4f})")
                
                lora_dropout_p = 0.0
                if hasattr(layer, "lora_dropout") and "default" in layer.lora_dropout:
                    lora_dropout_p = layer.lora_dropout["default"].p

                resize_lora_layer_svd(
                    layer=layer,
                    new_rank=new_rank,
                    lora_alpha=self.lora_alpha, # Using global alpha for consistency
                    adapter_name="default",
                    lora_dropout=lora_dropout_p,
                    **update_kwargs
                )
            else:
                if self.verbose:
                    print(f"  - {name}: r={new_rank} (Unchanged, Score: {score:.4f})")

        self.latest_ranks = new_ranks
        if self.verbose: print(f"✅ AdaptiveLoRA: Rank setup complete.\n")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs
    ):
        epoch = int(state.epoch) if state.epoch is not None else -1
        if self.latest_ranks and self.latest_scores:
            save_epoch_log(self.log_file, epoch, self.latest_ranks, self.latest_scores)
            if self.verbose:
                print(f"📄 Epoch {epoch}: Rank allocations logged to {self.log_file}\n")