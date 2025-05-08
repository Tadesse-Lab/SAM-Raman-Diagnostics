from pathlib import Path
import torch
import numpy as np
import torch.nn.functional as F
from utils.eval_helpers import evaluate
from utils.data_helpers import save_json
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)
    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction="none").sum(-1)


def train(
    output_dir: Path,
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    epochs: int = 200,
    early_stopping_patience: int = 15,
    label_smoothing: float = 0.1,
) -> Path:
    training_dir = output_dir / "train"
    training_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    early_stopping_counter = 0
    best_val_accuracy = 0.0
    best_val_epoch = 0
    best_model_path = training_dir / "best_val_model.pth"

    is_sam_optimizer = hasattr(optimizer, "first_step") and hasattr(
        optimizer, "second_step"
    )

    for epoch_idx in range(epochs):
        model.train()
        batch_losses = []
        batch_accuracies = []

        for inputs, targets in dataset.train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if is_sam_optimizer:
                # run first step
                enable_running_stats(model)
                pred = model(inputs)

                loss = smooth_crossentropy(pred, targets, smoothing=label_smoothing)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                # second step
                disable_running_stats(model)
                smooth_crossentropy(model(inputs), targets, smoothing=label_smoothing)
                optimizer.second_step(zero_grad=True)
            else:
                pred = model(inputs)
                loss = smooth_crossentropy(pred, targets, smoothing=label_smoothing)
                loss.mean().backward()
                optimizer.step()

            with torch.no_grad():
                correct = torch.argmax(pred.data, 1) == targets
                accuracy = correct.float().mean().item()
                loss_avg = loss.mean().item()

                batch_losses.append(loss_avg)
                batch_accuracies.append(accuracy)

        train_loss = np.mean(batch_losses)
        train_accuracy = np.mean(batch_accuracies)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        val_accuracy = evaluate(
            model=model, dataloader=dataset.val_loader, device=device
        )

        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_epoch = epoch_idx
            torch.save(
                {
                    "epoch": best_val_epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": best_val_accuracy,
                },
                best_model_path,
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        print(
            f"epoch {epoch_idx} train_loss: {train_loss} val_acc:{val_accuracy} best_val_epoch: {best_val_epoch} learning_rate: {current_lr}"
        )

        if early_stopping_counter >= early_stopping_patience:
            break

    results_path = training_dir / "results.json"
    results = {
        "train_loss": train_losses,
        "train_acc": train_accuracies,
        "val_acc": val_accuracies,
        "best_val_epoch": best_val_epoch,
    }
    save_json(results_path, results)

    return best_model_path
