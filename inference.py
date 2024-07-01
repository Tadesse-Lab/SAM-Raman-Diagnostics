import argparse
import torch
import numpy as np
import time
import os

from utils.smooth_cross_entropy import smooth_crossentropy
from utils.log import Log

from resnet_1d import ResNet
from dataset import RamanSpectra

import sys; sys.path.append("..")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectra_dir', nargs='+', default=['data/spectral_data/X_2018clinical.npy', 'data/spectral_data/X_2019clinical.npy'], help='Directory to spectra.')
    parser.add_argument('--label_dir', nargs='+', default=['data/spectral_data/y_2018clinical.npy', 'data/spectral_data/y_2019clinical.npy'], help='Directory to labels.')
    parser.add_argument('--spectra_interval', nargs='+', type=int, default=[400, 100], help='Specified patient intervals for clinical significance.')
    parser.add_argument('--weight_dir', default=None, type=str, help='Directory containing model weight(s).')
    parser.add_argument('--param_dir', default=None, type=str, help='Directory containing model parameters.')
    parser.add_argument('--seed', default=None, type=int, help='Initialization seed.')
    parser.add_argument('--shuffle', default=None, type=bool, help='Shuffle training set.')
    parser.add_argument('--save', action='store_true', help='Save results.')
    args = parser.parse_args()

    params = np.load(args.param_dir, allow_pickle=True).tolist()

    if args.spectra_dir and args.label_dir and args.spectra_interval:
        spectra_dir = args.spectra_dir
        label_dir = args.label_dir
        spectra_interval = args.spectra_interval
    else:
        spectra_dir = params['spectra_dir']
        label_dir = params['label_dir']
        spectra_interval = params['spectra_interval']

    seed = args.seed if args.seed is not None else params['seed']
    shuffle = args.shuffle if args.shuffle is not None else params['shuffle']

    dir = f"{params['optimizer']}{'_' + params['base_optimizer'] if params['optimizer'] in ['SAM', 'ASAM'] else ''}_{seed}"

    if args.save:
        count = 1
        unique_dir = f'outputs/{dir}_{count}'
        
        while os.path.exists(unique_dir):
            count += 1
            unique_dir = f'outputs/{dir}_{count}'
        
        os.makedirs(unique_dir, exist_ok=True)

    dataset = RamanSpectra(spectra_dir, label_dir, spectra_interval, seed, shuffle, num_workers=2, 
                           batch_size=params['batch_size'])
    log = Log(log_each=10)

    hidden_sizes = [params['hidden_size']] * params['layers']
    num_blocks = [params['block_size']] * params['layers']

    model = ResNet(hidden_sizes, num_blocks, input_dim=params['input_dim'],
                in_channels=params['in_channels'], num_classes=params['num_classes'], activation=params['activation'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = smooth_crossentropy

    model.load_state_dict(torch.load(args.weight_dir))
    
    model.eval()

    batch_loss = []
    batch_acc = []

    start_time = time.time()
    with torch.no_grad():
        for batch in dataset.test:
            inputs, targets = (b.to(device) for b in batch)
            inputs = inputs.float()

            predictions = model(inputs)
            targets = targets.to(torch.long)
            loss = smooth_crossentropy(predictions, targets)
            correct = torch.argmax(predictions, 1) == targets
            accuracy = correct.float().mean().item()
            loss_avg=loss.mean().item()

            batch_loss.append(loss_avg)
            batch_acc.append(accuracy)

    end_time = time.time()
    inference_time = end_time - start_time

    test_loss = np.mean(batch_loss)
    test_accuracy = np.mean(batch_acc)

    print('Test Loss: ', test_loss, 'Test Acc: ', test_accuracy)

    scores = {
        'test-time': inference_time,
        'test-loss': test_loss,
        'test-acc': test_accuracy,
    }

    if args.save:
        print(f'Saving values at {unique_dir}')
        np.save(f'{unique_dir}/scores.npy', scores)
