import torch.nn as nn
from typing import List
from models.resnet_1d import ResNet


def get_model(
    model_name: str,
    hidden_sizes: List[int],
    num_blocks: List[int],
    input_dim: int,
    in_channels: int,
    num_classes: int,
    activation: str = "selu",
) -> nn.Module:
    """Retrieves and initializes select model architecture.

    Args:
        model_name: name of desired model to initialize
        hidden_sizes: list of hidden layer sizes
        num_blocks: list of block sizes per layer
        input_dim: input dimension of model
        in_channels: number of input channels
        num_classes: number of output classes
        activation: activation function to use

    Returns:
        Initialization of model
    """
    if model_name.lower() == "resnet":
        return ResNet(
            hidden_sizes=hidden_sizes,
            num_blocks=num_blocks,
            input_dim=input_dim,
            in_channels=in_channels,
            num_classes=num_classes,
            activation=activation,
        )
    else:
        raise ValueError(f"Model {model_name} not supported. Choose from 'resnet'.")
