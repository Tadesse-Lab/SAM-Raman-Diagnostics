import torch
import torch.nn as nn
from utils.sam import SAM


def get_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    base_optimizer_name: str = "adam",
    rho: float = 0.05,
    weight_decay: float = 0.0005,
) -> torch.optim:
    """Retrieves and initializes select optimizer.

    Args:
        optimizer_name: name of target optimizer to initialize
        model: the model parameters to be optimized
        learning_rate: learning rate of optimizer
        base_optimizer_name: base optimizer for sam optimizers
        rho: size of rho step
        weight_decay: desired weight decay for optimizer
    """
    if optimizer_name == "adam":
        print("Using Adam optimizer")
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, betas=(0.5, 0.999)
        )
    elif optimizer_name == "sgd":
        print("Using SGD optimizer")
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name in ["sam", "asam"]:
        if base_optimizer_name == "adam":
            base_optim = torch.optim.Adam
        elif base_optimizer_name == "sgd":
            base_optim = torch.optim.SGD
        else:
            raise ValueError(
                f"Invalid base optimizer: {base_optimizer_name}. Choose from 'adam' or 'sgd'."
            )
        if optimizer_name == "sam":
            print("Using SAM optimizer")
            adaptive = False
        else:
            print("Using ASAM optimizer")
            adaptive = True
        return SAM(
            model.parameters(),
            base_optim,
            rho=rho,
            adaptive=adaptive,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"Invalid optimizer: {optimizer_name}. Choose from 'adam', 'sgd', 'sam', or 'asam'."
        )


def get_scheduler(
    scheduler_name: str, optimizer: torch.optim, epochs: int, lr_step: float = 0.1
) -> torch.optim.lr_scheduler:
    """Retrieves and initializes select learning rate scheduler.

    Args:
        scheduler_name: name of scheduler to initialize
        optimizer: The optimizer whose learning rate will be scheduled
        epochs: number of epochs used during training
        lr_step: step size of learning rate

    Returns:
        Initialization of learning rate scheduler, or None if no scheduler is specified
    """
    if scheduler_name == "step":
        print("Using StepLR")
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=(epochs // 4), gamma=lr_step
        )
    elif scheduler_name == "cosine":
        print("Using CosineAnnealingLR")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(epochs // 4), eta_min=0
        )
    else:
        print("No LR scheduler")
        return None
