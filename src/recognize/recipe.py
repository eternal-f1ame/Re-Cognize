"""The training recipe: one place, imported by the CLI defaults, the launcher and the tests.
Changing a value here is a protocol change."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class Recipe:
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    batch_size: int = 32              # only used without PK sampling (evaluation-side batching)
    p: int = 8
    k: int = 4
    k_support: int = 2
    id_drop: float = 0.5
    proto_temperature: float = 0.15
    triplet_margin: float = 0.3
    w_proto: float = 1.0
    w_trip: float = 1.0
    w_mem: float = 0.1
    w_ce: float = 0.3                 # 0 in memory mode (launcher)
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_layers: int = 4
    lora_lr: float = 1e-5
    memory_lr_scale: float = 1.0
    working_capacity: int = 8
    slots_per_char: int = 5
    num_heads: int = 8
    dropout: float = 0.1
    seed: int = 0
    save_every: int = 10
    dev_eval_every: int = 10
    amp: bool = True

    def as_dict(self) -> Dict:
        return asdict(self)


RECIPE = Recipe()

# CLI destination name -> recipe field, for the defaults test and the launcher.
CLI_FIELDS = {
    "epochs": "epochs", "lr": "lr", "weight_decay": "weight_decay", "warmup_epochs": "warmup_epochs",
    "p": "p", "k": "k", "k_support": "k_support", "episodic_id_drop_rate": "id_drop",
    "proto_temperature": "proto_temperature", "triplet_margin": "triplet_margin",
    "prototype_weight": "w_proto", "triplet_weight": "w_trip", "memory_weight": "w_mem", "ce_weight": "w_ce",
    "lora_rank": "lora_rank", "lora_alpha": "lora_alpha", "lora_layers": "lora_layers", "lora_lr": "lora_lr",
    "memory_lr_scale": "memory_lr_scale", "working_capacity": "working_capacity", "slots_per_char": "slots_per_char",
    "seed": "seed", "save_freq": "save_every", "val_freq": "dev_eval_every", "amp": "amp",
}
