import json
from typing import Dict
from pathlib import Path
import torch
import numpy as np
import random


def save_json(file_path: Path, data: Dict) -> None:
    """Save data results as JSON.

    Args:
        file_path: path to save data
        data: results to save
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def set_all_seeds(seed):
    """Set all seeds to make results reproducible.

    Args:
        seed: desired seed to set
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
