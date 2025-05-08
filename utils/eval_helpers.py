import torch
from torch.utils.data import DataLoader
import numpy as np


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    model.to(device)
    model.eval()

    batch_acc = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)

            correct = torch.argmax(predictions, 1) == targets
            accuracy = correct.float().mean().item()

            batch_acc.append(accuracy)

    avg_accuracy = np.mean(batch_acc)

    return avg_accuracy
