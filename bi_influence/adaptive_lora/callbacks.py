import os
import logging
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from .importance import compute_bi_scores
from .allocation import allocate_ranks_bi
from .utils import get_lora_layers, save_epoch_log

logger = logging.getLogger(__name__)

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
        lora_alpha_multiplier:int=4
    ):
        self.lora_alpha_multiplier=lora_alpha_multiplier
        self.total_rank = total_rank
        self.val_dataloader = val_dataloader
        self.tau = tau
        self.min_rank = min_rank
        self.verbose = verbose
        self.validate_batch_size = validate_batch_size
        self.log_file = os.path.join(log_path, "adaptive_lora_epoch_logs.csv")

        os.makedirs(log_path, exist_ok=True)

        self.latest_scores = {}
        self.latest_ranks = {}


    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs
    ):
        epoch = int(state.epoch) + 1 if state.epoch is not None else 0

        if self.verbose:
            print(f"\n--- AdaptiveLoRA: Preparing ranks for Epoch {epoch} ---")

        device = next(model.parameters()).device

        if self.verbose:
            print("Computing BI importance scores (pre-training)...")
        scores = compute_bi_scores(
                model,
                self.val_dataloader,
                device,
                batch_size=self.validate_batch_size,
        )

        if not scores:
            if self.verbose:
                print("⚠️ No LoRA layers or BI scores found. Skipping rank update.")
            return

        if self.verbose:
            print("Allocating new ranks based on BI scores...")
        new_ranks = allocate_ranks_bi(scores, self.total_rank, self.tau,self.min_rank)

        if self.verbose:
            print("Applying new ranks to LoRA modules for this epoch...")

        lora_layers = get_lora_layers(model)
        config = model.peft_config.get("default")
        if not config:
            logger.error("❌ PEFT config not found. Skipping update.")
            return

        init_lora_weights = getattr(config, "init_lora_weights", True)
        use_rslora = getattr(config, "use_rslora", False)
        use_dora = getattr(config, "use_dora", False)
        use_qalora = getattr(config, "use_qalora", False)
        lora_bias = getattr(config, "bias", "none")
        qalora_group_size = getattr(config, "qalora_group_size", 64)

        for name, layer in lora_layers.items():
            new_rank = new_ranks.get(name)
            if new_rank is None:
                continue

            current_rank = layer.r.get("default", 0)
            score = scores.get(name, 0.0)

            if current_rank != new_rank:
                if self.verbose:
                    print(f"  - {name}: r={current_rank} → {new_rank} (Score: {score:.4f})")
            else:
                if self.verbose:
                    print(f"  - {name}: r={new_rank} (Unchanged, Score: {score:.4f})")

            if hasattr(layer, "update_layer") and current_rank != new_rank:
                lora_dropout_p = 0.0
                if hasattr(layer, "lora_dropout") and "default" in layer.lora_dropout:
                    lora_dropout_p = layer.lora_dropout["default"].p

                layer.update_layer(
                    adapter_name="default",
                    r=new_rank,
                    lora_alpha=new_rank*self.lora_alpha_multiplier,
                    lora_dropout=lora_dropout_p,
                    init_lora_weights=init_lora_weights,
                    use_rslora=use_rslora,
                    use_dora=use_dora,
                    use_qalora=use_qalora,
                    lora_bias=lora_bias,
                    qalora_group_size=qalora_group_size,
                )

        self.latest_scores = scores
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
