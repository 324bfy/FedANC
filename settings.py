from datetime import datetime
from typing import List

defaults = {
    "available_models": ["basic", "hybrid_a", "hybrid_b"],
    "active_model": "basic",
    "debug_flag": False,
    "use_tensorboard": False,

    "train_config": {
        "run_tag": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "batch_num": 256,
        "epoch_num": 50
    },

    "basic": {
        "block_dim": 16,
        "batch_size": 64,
        "lr_comm": 0.001,
        "lr_adv": 0.001,
    },

    "hybrid_a": {
        "block_dim": 16,
        "batch_size": 64,
        "lr_comm": 0.001,
        "lr_adv": 0.001,
    },

    "hybrid_b": {
        "block_dim": 16,
        "batch_size": 64,
        "lr_comm": 0.001,
        "lr_adv": 0.001,
    },

    "federated": {
        "client_count": 2,
        "comm_rounds": 20,
        "local_epochs": 5,
        "local_batches": 256,
        "encrypt_exchange": False,
    },

    "save_checkpoint": True,
}

def update_config(cli_args: List[str]):
    global defaults
    cfg = defaults
    for arg in cli_args:
        if "=" not in arg:
            continue
        key, val = arg.split("=")
        parts = key.split("-")
        target = cfg
        for p in parts[:-1]:
            target = target[p]
        if isinstance(target[parts[-1]], dict):
            continue
        orig_type = type(target[parts[-1]])
        target[parts[-1]] = orig_type(val)