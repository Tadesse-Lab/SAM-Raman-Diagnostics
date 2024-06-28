import torch
from sam import SAM

def get_optimizer(model, optimizer_name, learning_rate, base_optimizer_name=None, rho=None, weight_decay=None):
    base_optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam
    }
    optimizers = {
        'Adam': lambda: torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999)),
        'SGD': lambda: torch.optim.SGD(model.parameters(), lr=learning_rate),
        'SAM': lambda: SAM(model.parameters(), base_optimizer, rho=rho, adaptive=False, lr=learning_rate, weight_decay=weight_decay),
        'ASAM': lambda: SAM(model.parameters(), base_optimizer, rho=rho, adaptive=True, lr=learning_rate, weight_decay=weight_decay)
    }
    if base_optimizer_name:
        if base_optimizer_name not in base_optimizers:
            raise ValueError(f'Unsupported base optimizer: {base_optimizer_name}')
        
        base_optimizer = base_optimizers[base_optimizer_name]
        print(f'Base Optimizer: {base_optimizer_name}')

    if optimizer_name not in optimizers:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')

    optimizer = optimizers[optimizer_name]()
    print(f'Optimizer: {optimizer_name}')

    return optimizer, base_optimizer
