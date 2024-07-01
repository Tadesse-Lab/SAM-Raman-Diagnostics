import argparse
import torch
import numpy as np
import time
import os

from utils.smooth_cross_entropy import smooth_crossentropy
from utils.log import Log
from utils.initialize import get_optimizer, get_scheduler
from utils.bypass_bn import enable_running_stats, disable_running_stats

from resnet_1d import ResNet
from dataset import RamanSpectra

import sys; sys.path.append("..")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int, help='Total number of epochs.')
    parser.add_argument('--spectra_dir', default=['data/spectral_data/X_2018clinical.npy', 'data/spectral_data/X_2019clinical.npy'], type=list, help='Directory to spectra.')
    parser.add_argument('--label_dir', default=['data/spectral_data/y_2018clinical.npy', 'data/spectral_data/y_2019clinical.npy'], type=list, help='Directory to labels.')
    parser.add_argument('--spectra_interval', default=[400, 100], type=list, help = 'Specified patient intervals for clinical significance.')

    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for the training and validation loops.')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate for the optimizer.')
    parser.add_argument('--in_channels', default=64, type=int, help='Number of input channels.')
    parser.add_argument('--layers', default=6, type=int, help='Number of layers in the model.')
    parser.add_argument('--hidden_size', default=100, type=int, help='Size of the hidden layer in the model.')
    parser.add_argument('--block_size', default=2, type=int, help='Block size for the model.')
    parser.add_argument('--num_classes', default=5, type=int, help='Number of classes in the classification task.')
    parser.add_argument('--input_dim', default=1000, type=int, help='Input dimension for the model.')
    parser.add_argument('--rho', default=0.05, type=float, help='Rho value for the optimizer.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for the optimizer.')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay value for the optimizer.')
    parser.add_argument('--label_smoothing', default=0.1, type=float, help='Use 0.0 for no label smoothing.')
    parser.add_argument('--activation', default='selu', type=str, choices=['relu','selu','gelu'], help='Activation function to use for ResNet.')
    parser.add_argument('--optimizer', default='SAM', type=str, choices=['SAM','ASAM','Adam', 'SGD'], help='Optimizer to be used.')
    parser.add_argument('--base_optimizer', default='SGD', type=str, choices=['Adam', 'SGD'], help='Base optimizer to be used.')
    parser.add_argument('--scheduler', default='step', type=str, choices=['step', 'cosine'], help='Learning rate scheduler to be used.')
    parser.add_argument('--lr_step', default=0.2, type=float, help='Step size for learning rate scheduler.')
    parser.add_argument('--seed', default=42, type=int, help='Initialization seed.')
    parser.add_argument('--shuffle', default=True, type=bool, help='Shuffle training set.')
    parser.add_argument('--save', action='store_true', help='Save results.')

    args = parser.parse_args()
    default_args = {action.dest: action.default for action in parser._actions if action.dest != 'help'}
    args_dict = {**default_args, **vars(args)}

    dir = f"{args.optimizer}{'_' + args.base_optimizer if args.optimizer in ['SAM', 'ASAM'] else ''}_{args.seed}"

    if args.save:
        count = 1
        unique_dir = f'train/{dir}_{count}'
        
        while os.path.exists(unique_dir):
            count += 1
            unique_dir = f'train/{dir}_{count}'
        
        os.makedirs(unique_dir, exist_ok=True)
        np.save(f'{unique_dir}/params.npy', args_dict)

    dataset = RamanSpectra(args.spectra_dir, args.label_dir, args.spectra_interval, args.seed, 
                            args.shuffle, num_workers=2, batch_size=args.batch_size)
    log = Log(log_each=10)

    hidden_sizes = [args.hidden_size] * args.layers
    num_blocks = [args.block_size] * args.layers

    model = ResNet(hidden_sizes, num_blocks, input_dim=args.input_dim,
                in_channels=args.in_channels, num_classes=args.num_classes, activation=args.activation)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = smooth_crossentropy

    optimizer, base_optimizer = get_optimizer(model, args.optimizer, args.learning_rate, args.base_optimizer, 
                                              args.rho, args.weight_decay)
    
    scheduler_optimizer = optimizer.base_optimizer if args.optimizer in ['SAM', 'ASAM'] else optimizer
    scheduler = get_scheduler(args.scheduler, scheduler_optimizer, args.epochs, args.lr_step)



    print('Starting Training')
    
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    time_epochs = []
    best_acc = 0

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        log.train(len_dataset=len(dataset.train))
        batch_loss = []
        batch_acc = []

        for train_index, batch in enumerate(dataset.train):
            inputs, targets = (b.to(device) for b in batch)
            inputs = inputs
            
            if args.optimizer not in ['SAM', 'ASAM']:
                predictions = model(inputs)
                targets = targets.to(torch.long)

                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)

                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
            else:
                enable_running_stats(model)
                predictions = model(inputs)
                targiets = targets.to(torch.long)
                loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                disable_running_stats(model)
                smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
                optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                accuracy = correct.float().mean().item()
                loss_avg = loss.mean().item()

                batch_loss.append(loss_avg)
                batch_acc.append(accuracy) 
                
                log(model, loss.cpu(), correct.cpu(), scheduler.get_last_lr()[0])

        epoch_loss_avg = np.mean(batch_loss)
        epoch_accuracy_avg = np.mean(batch_acc)
        train_loss.append(epoch_loss_avg)
        train_accuracy.append(epoch_accuracy_avg)

        scheduler.step()

        model.eval()
        log.eval(len_dataset=len(dataset.val))

        batch_loss = []
        batch_acc = []

        with torch.no_grad():
            for batch in dataset.val:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                targets = targets.to(torch.long)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                accuracy = correct.float().mean().item()
                
                try:
                    loss_avg = loss.mean().item()
                except:
                    loss_avg = 0

                batch_loss.append(loss_avg)
                batch_acc.append(accuracy)

                log(model, loss.cpu(), correct.cpu())

        epoch_loss_avg = np.mean(batch_loss)
        epoch_accuracy_avg = np.mean(batch_acc)
        val_loss.append(epoch_loss_avg)
        val_accuracy.append(epoch_accuracy_avg)

        end_time = time.time()
        epoch_time = end_time - start_time
        time_epochs.append(epoch_time)

        if epoch_accuracy_avg > best_acc and args.save:
            best_val = epoch_accuracy_avg
            torch.save(model.state_dict(), f'{unique_dir}/best_val.pth')

        log.flush

    scores = {
        'train-time': time_epochs,
        'train-loss': train_loss,
        'train-acc': train_accuracy,
        'val-loss': val_loss,
        'val-acc': val_accuracy,
    }

    if args.save:
        print(f'Saving values at {unique_dir}')
        np.save(f'{unique_dir}/scores.npy', scores)
