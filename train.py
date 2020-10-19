import argparse
import random
import datetime
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import models
import config as cfg

def train(report, optimizer, log_interval, epoch,  device, model, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        report.add_scalar('train_mse', loss)

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Epoch: %4d\tBatch: %d\tTraining Loss: %.4f' % (epoch, batch_idx, loss.item()))
        
        report.advance_step()

def evaluate(epoch, device, model, val_loader):
    model.eval()
    loss = 0.0
    batches = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.mse_loss(output, target)
            batches += 1
    loss = loss / batches
    print('Epoch: %4d\tValidatetion loss: %.4f' % (epoch, loss.item()))
    return loss

def output_dirs(config):
    name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    base_dir = config['output_path']
    out_dir = os.path.join(base_dir, name)
    pred_dir = os.path.join(out_dir, 'preds')
    log_dir = os.path.join(out_dir, 'log_dir')
    for dir in [base_dir, out_dir, pred_dir, log_dir]:
        os.makedirs(dir, exist_ok=True)
    return out_dir, pred_dir, log_dir

def create_waveunet_model(config):
    window_sizes, model = models.waveunet(config['output_size'], 2, 2, config['down_kernel_size'], config['up_kernel_size'], config['depth'], config['num_filters'])
    print('Creating WaveUNet model with input output sample size: {}'.format(window_sizes))
    return window_sizes, model

def main():
    parser = argparse.ArgumentParser(description='Source Separation Trainer')
    parser.add_argument('--config', help='Path to config file', required=True)
    args = parser.parse_args()
    
    config = cfg.load(args.config)
    window_size, model = create_waveunet_model(config)
    
    out_dir, pred_dir, log_dir = output_dirs(config)
    cfg.save(os.path.join(out_dir, 'config.yml'), config)
    print('Saving output to directory %s' % out_dir)

    device = torch.device(config['device'])
    model = model.to(device)
    print(model)

if __name__ == '__name__':
    main()