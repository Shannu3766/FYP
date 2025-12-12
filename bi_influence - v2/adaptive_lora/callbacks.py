import os
import logging
import torch
from peft.tuners.lora import LoraLayer
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from .importance import compute_bi_scores
from .allocation import allocate_ranks_bi
from .utils import get_lora_layers, save_epoch_log

logger = logging.getLogger(__name__)

def resize_lora_layer_svd(
    layer: LoraLayer, 
    new_rank: int, 
    lora_alpha: int, 
    adapter_name: str = "default",
    **kwargs
):
    with torch.no_grad():
        if adapter_name not in layer.lora_A:
            return
        old_r = layer.r[adapter_name]        
        if old_r == 0: 
             layer.update_layer(adapter_name, new_rank, lora_alpha=lora_alpha, init_lora_weights=True, **kwargs)
             return

        old_alpha = layer.lora_alpha[adapter_name]
        old_scaling = old_alpha / old_r

        A_old = layer.lora_A[adapter_name].weight
        B_old = layer.lora_B[adapter_name].weight
        

        W_delta = (B_old @ A_old) * old_scaling


        dtype = A_old.dtype
        U, S, Vh = torch.linalg.svd(W_delta.float(), full_matrices=False)
        
        k = new_rank
        k = min(k, S.size(0))
        
        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]
        

        sqrt_S = torch.diag(torch.sqrt(S_k))
        B_new = (U_k @ sqrt_S).to(dtype)
        A_new = (sqrt_S @ Vh_k).to(dtype)

        if new_rank > 0:
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
        device = layer.lora_A[adapter_name].weight.device
        
        if k < new_rank:
             layer.lora_A[adapter_name].weight.data.zero_()
             layer.lora_B[adapter_name].weight.data.zero_()
             layer.lora_A[adapter_name].weight.data[:k, :] = A_new.to(device)
             layer.lora_B[adapter_name].weight.data[:, :k] = B_new.to(device)
        else:
             layer.lora_A[adapter_name].weight.data = A_new.to(device)
             layer.lora_B[adapter_name].weight.data = B_new.to(device)


class AdaptiveLoRACallback(TrainerCallback):
    def __init__(
        self,
        total_rank: int,
        val_dataloader,
        min_rank: int = 4,
        tau: float = 1.0,
        log_path: str = "./logs",
        verbose: bool = True,
        validate_batch_size: int = 4,
        lora_alpha_multiplier: int = 4,
        score_smoothing_beta: float = 0.0,
        update_interval: int = 1,
        warmup_epochs: int = 0,
        cooldown_epochs: int = 0
    ):
        self.total_rank = total_rank
        self.val_dataloader = val_dataloader
        self.min_rank = min_rank
        self.tau = tau
        self.verbose = verbose
        self.validate_batch_size = validate_batch_size
        self.lora_alpha_multiplier = lora_alpha_multiplier        
        self.score_smoothing_beta = score_smoothing_beta
        self.update_interval = update_interval
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.log_file = os.path.join(log_path, "adaptive_lora_epoch_logs.csv")
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
        total_epochs = args.num_train_epochs        
        if epoch <= self.warmup_epochs:
            if self.verbose:
                print(f"⏳ Warmup Period ({epoch}/{self.warmup_epochs}). Skipping rank update.")
            return
        if epoch > (total_epochs - self.cooldown_epochs):
            if self.verbose:
                print(f"❄️ Cooldown Period ({epoch} > {total_epochs - self.cooldown_epochs}). Skipping rank update.")
            return
        if (epoch - self.warmup_epochs - 1) % self.update_interval != 0:
            if self.verbose:
                print(f"⏩ Skipping update (Interval={self.update_interval}).")
            return
        device = next(model.parameters()).device
        if self.verbose:
            print("Computing BI importance scores...")
        current_scores = compute_bi_scores(
                model,
                self.val_dataloader,
                device,
                batch_size=self.validate_batch_size,
        )

        if not current_scores:
            if self.verbose:
                print("⚠️ No LoRA layers or BI scores found. Skipping.")
            return

        if self.score_smoothing_beta > 0.0:
            if not self.ema_scores:
                self.ema_scores = current_scores
            else:
                # Update EMA: S_t = beta * S_{t-1} + (1-beta) * S_current
                for name, score in current_scores.items():
                    prev_score = self.ema_scores.get(name, score)
                    self.ema_scores[name] = (self.score_smoothing_beta * prev_score) + \
                                            ((1 - self.score_smoothing_beta) * score)
            scores_to_use = self.ema_scores
            if self.verbose: print(f"📊 Applied Score Smoothing (beta={self.score_smoothing_beta})")
        else:
            scores_to_use = current_scores

        self.latest_scores = scores_to_use

        if self.verbose:
            print("Allocating new ranks...")
        new_ranks = allocate_ranks_bi(scores_to_use, self.total_rank, self.tau, self.min_rank)

        if self.verbose:
            print("Applying SVD rank updates to LoRA modules...")

        lora_layers = get_lora_layers(model)
        config = model.peft_config.get("default")
        if not config:
            logger.error("❌ PEFT config not found. Skipping update.")
            return

        update_kwargs = {
            "use_rslora": getattr(config, "use_rslora", False),
            "use_dora": getattr(config, "use_dora", False),
            "use_qalora": getattr(config, "use_qalora", False),
            "lora_bias": getattr(config, "bias", "none"),
            "qalora_group_size": getattr(config, "qalora_group_size", 64),
            "lora_dropout": getattr(config, "lora_dropout", 0.0)
        }

        for name, layer in lora_layers.items():
            new_rank = new_ranks.get(name)
            if new_rank is None:
                continue

            current_rank = layer.r.get("default", 0)
            score = scores_to_use.get(name, 0.0)

            if current_rank != new_rank:
                if self.verbose:
                    print(f"  - {name}: r={current_rank} → {new_rank} (Score: {score:.4f})")                
                resize_lora_layer_svd(
                    layer=layer,
                    new_rank=new_rank,
                    lora_alpha=new_rank * self.lora_alpha_multiplier,
                    adapter_name="default",
                    **update_kwargs
                )
            else:
                if self.verbose:
                    print(f"  - {name}: r={new_rank} (Unchanged, Score: {score:.4f})")

        self.latest_ranks = new_ranks
        if self.verbose:
            print(f"✅ AdaptiveLoRA: Rank setup for Epoch {epoch} complete.\n")

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
                print(
                    f"📄 Epoch {epoch}: Rank allocations logged to {self.log_file}\n"
                )