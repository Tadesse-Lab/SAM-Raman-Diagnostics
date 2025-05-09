import argparse
import torch
from pathlib import Path

from utils.train_helpers import train
from utils.eval_helpers import evaluate
from utils.dataset import SAMRaman
from utils.optimizer import get_optimizer, get_scheduler
from utils.config import Config
from utils.data_helpers import save_json, set_all_seeds

from models.model_factory import get_model


def parse_args():
    config = Config()
    config_dict = config.to_dict()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spectra_dirs",
        nargs="+",
        type=str,
        help=f"Directories containing spectral data files. Default: {config.spectra_dirs}",
    )
    parser.add_argument(
        "--label_dirs",
        nargs="+",
        type=str,
        help=f"Directories containing label data files. Default: {config.label_dirs}",
    )
    parser.add_argument(
        "--spectra_intervals",
        nargs="+",
        type=int,
        help=f"Specified patient intervals for clinical significance. Default: {config.spectra_intervals}",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help=f"Batch size for the training and validation loops. Default: {config.batch_size}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help=f"Total number of training epochs. Default: {config.epochs}",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help=f"Initial learning rate for training. Default: {config.learning_rate}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=f"Seed to use for reproducibility. Default: {config.seed}",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet1d"],
        help=f"Model architecture to be used for training. Default: {config.model}",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        help=f"Number of input channels. Default: {config.in_channels}",
    )
    parser.add_argument(
        "--layers",
        type=int,
        help=f"Number of layers in the model. Default: {config.layers}",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        help=f"Size of the hidden layer in the model. Default: {config.hidden_size}",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        help=f"Block size for the model. Default: {config.block_size}",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        help=f"Number of classes in the classification task. Default: {config.num_classes}",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        help=f"Input dimension for the model. Default: {config.input_dim}",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "selu", "gelu"],
        help=f"Activation function to use for model. Default: {config.activation}",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["sam", "asam", "adam", "sgd"],
        help=f"Optimizer to use for training. Default: {config.optimizer}",
    )
    parser.add_argument(
        "--base_optimizer",
        type=str,
        choices=["adam", "sgd"],
        help=f"Base optimizer for SAM/ASAM. Default: {config.base_optimizer}",
    )
    parser.add_argument(
        "--rho",
        type=float,
        help=f"Rho value for SAM/ASAM optimizers. Default: {config.rho}",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help=f"Weight decay for optimizer. Default: {config.weight_decay}",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        help=f"Momentum value for SGD. Default: {config.momentum}",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        help=f"Label smoothing factor. Default: {config.label_smoothing}",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["step", "cosine", "none"],
        help=f"Learning rate scheduler to use. Default: {config.scheduler}",
    )
    parser.add_argument(
        "--lr_step",
        type=float,
        help=f"Learning rate step size. Default: {config.lr_step}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help=f"Output directory to save training results. Default: {config.output_dir}",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help=f"Name of experiment. Default: {config.experiment_name}",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help=f"Number of epochs without improvement for early stopping. Default: 15",
    )

    args = parser.parse_args()
    args_dict = vars(args)

    # update default config values with passed results
    parsed_args = {k: v for k, v in args_dict.items() if v is not None}
    config_dict.update(parsed_args)

    return config_dict


def run_model_training_and_evaluation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = parse_args()
    set_all_seeds(config["seed"])

    experiment_dir = Path(config["output_dir"]) / config["experiment_name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    save_json(experiment_dir / "experimental_params.json", config)

    dataset = SAMRaman(
        spectral_dirs=config["spectra_dirs"],
        label_dirs=config["label_dirs"],
        patient_intervals=config["spectra_intervals"],
        batch_size=config["batch_size"],
        seed=config["seed"],
    )

    # create model
    hidden_sizes = [config["hidden_size"]] * config["layers"]
    num_blocks = [config["block_size"]] * config["layers"]
    model = get_model(
        model_name=config["model"],
        hidden_sizes=hidden_sizes,
        num_blocks=num_blocks,
        input_dim=config["input_dim"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        activation=config["activation"],
    )

    # get optimizer and scheduler
    optimizer = get_optimizer(
        optimizer_name=config["optimizer"],
        model=model,
        learning_rate=config["learning_rate"],
        base_optimizer_name=config["base_optimizer"],
        rho=config["rho"],
        weight_decay=config["weight_decay"],
    )
    sched_optimizer = (
        optimizer.base_optimizer
        if config["optimizer"] in ["sam", "asam"]
        else optimizer
    )
    scheduler = get_scheduler(
        scheduler_name=config["scheduler"].lower(),
        optimizer=sched_optimizer,
        epochs=config["epochs"],
        lr_step=config["lr_step"],
    )

    best_model_path = train(
        output_dir=experiment_dir,
        dataset=dataset,
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=config["epochs"],
        early_stopping_patience=config["early_stopping_patience"],
        label_smoothing=config["label_smoothing"],
    )

    # evaluate on test set
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dir = experiment_dir / "inference"
    test_dir.mkdir(parents=True, exist_ok=True)

    test_accuracy = evaluate(model=model, dataloader=dataset.test_loader, device=device)

    print(f"test results -  test_acc: {test_accuracy}")

    results = {"test_acc": test_accuracy}
    save_json(test_dir / "test_results.json", results)

    print(f"Results saved to {experiment_dir}")


if __name__ == "__main__":
    run_model_training_and_evaluation()
