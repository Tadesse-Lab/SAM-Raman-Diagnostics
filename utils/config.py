from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class Config:
    # dataset params
    spectra_dirs: List[str] = field(
        default_factory=lambda: [
            "data/spectral_data/X_2018clinical.npy",
            "data/spectral_data/X_2019clinical.npy",
        ]
    )
    label_dirs: List[str] = field(
        default_factory=lambda: [
            "data/spectral_data/y_2018clinical.npy",
            "data/spectral_data/y_2019clinical.npy",
        ]
    )
    spectra_intervals: List[int] = field(default_factory=lambda: [400, 100])

    # training params
    batch_size: int = 16
    epochs: int = 200
    learning_rate: float = 0.001
    seed: int = 42

    # model params
    model: str = "resnet"
    in_channels: int = 64
    layers: int = 6
    hidden_size: int = 100
    block_size: int = 2
    num_classes: int = 5
    input_dim: int = 1000
    activation: str = "selu"

    # optimizer params
    optimizer: str = "sam"
    base_optimizer: str = "adam"
    rho: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0005
    label_smoothing: float = 0.1

    # scheduler params
    scheduler: str = "step"
    lr_step: float = 0.2

    # output params
    output_dir: str = "results"
    experiment_name: str = "sam-raman"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
