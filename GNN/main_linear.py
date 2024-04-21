import argparse
import os
import sys
import logging
import torch
import time
from model import GCNMLP, ModelArgs
from dataset import *
from utils import *
import wandb
import yaml
import math
from torch_geometric.loader import DataLoader

print = logging.info


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq',
                        type=int,
                        default=2,
                        help='print frequency')
    parser.add_argument('--save_freq',
                        type=int,
                        default=50,
                        help='save frequency')

    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='batch_size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs',
                        type=int,
                        default=300,
                        help='number of training epochs')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay_rate',
                        type=float,
                        default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--trial',
                        type=str,
                        default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--data_folder',
                        type=str,
                        default='./data',
                        help='path to custom dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='ESOL',
                        choices=['ESOL'],
                        help='dataset')
    parser.add_argument('--resume',
                        type=str,
                        default='',
                        help='resume ckpt path')
    parser.add_argument('--aug',
                        type=str,
                        default='crop,flip,color,grayscale',
                        help='augmentations')

    parser.add_argument('--ckpt',
                        type=str,
                        default='',
                        help='path to the trained encoder')

    parser.add_argument('--loss',
                        type=str,
                        default='L1',
                        choices=['L1', 'MSE', 'huber'],
                        help='loss function to train')
    
    parser.add_argument('--freeze_encoder',
                        default=False,
                        action='store_true',
                        help='whether or not to freeze the encoder trained with RnC.')

    opt = parser.parse_args()
    
    assert opt.ckpt != '', "Please provide a checkpoint for the model trained using RnC loss."

    opt.model_name = 'Regressor_{}_{}_ep_{}_lr_{}_d_{}_wd_{}_mmt_{}_bsz_{}_trial_{}_encoder_frozen_{}'. \
        format(opt.loss, opt.dataset, opt.epochs, opt.learning_rate, opt.lr_decay_rate,
               opt.weight_decay, opt.momentum, opt.batch_size, opt.trial, opt.freeze_encoder)
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-1][:-len('_last.pth')]
    opt.save_folder = '/'.join(opt.ckpt.split('/')[:-1])
    
    opt.experiment_name = "L1-GNNFull"

    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s",
                        handlers=[
                            logging.FileHandler(
                                os.path.join(opt.save_folder,
                                             f'{opt.model_name}.log')),
                            logging.StreamHandler()
                        ])

    print(f"Model name: {opt.model_name}")
    print(f"Options: {opt}")

    return opt


def set_loader(opt):

    train_dataset = globals()[opt.dataset](data_folder=opt.data_folder,
                                           split='train')
    val_dataset = globals()[opt.dataset](data_folder=opt.data_folder,
                                         split='valid')
    test_dataset = globals()[opt.dataset](data_folder=opt.data_folder,
                                          split='test')

    print(f'Train set size: {train_dataset.__len__()}\t'
          f'Val set size: {val_dataset.__len__()}\t'
          f'Test set size: {test_dataset.__len__()}')

    train_loader = DataLoader(train_dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers,
                            pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            pin_memory=True)

    return train_loader, val_loader, test_loader


def set_model(opt):
    model_args = ModelArgs() # HARDCODED FIXME
    model_args.in_channels = 9
    model_args.out_channels = 100
    model_args.hidden_channels = 64
    model_args.num_layers = 5
    model_args.dropout = 0.0
    model = GCNMLP(model_args)
    if opt.freeze_encoder:
        print("Freezing the encoder module.")
        for param in model.encoder.parameters():
            param.requires_grad = False
    if opt.loss == 'L1':
        criterion = torch.nn.L1Loss()
    elif opt.loss == 'MSE':
        criterion = torch.nn.MSELoss()
    elif opt.loss == 'huber':
        criterion = torch.nn.SmoothL1Loss()
    else:
        raise ValueError(f"Loss function {opt.loss} not supported!")
    
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.device_count() > 1:
        model.encoder = torch.nn.DataParallel(model.encoder)
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
    model = model.cuda()
    criterion = criterion.cuda()
    torch.backends.cudnn.benchmark = True

    model.load_state_dict(state_dict, 
                          strict=False) # Load only RnC encoder
    print(f"<=== Epoch [{ckpt['epoch']}] checkpoint Loaded from {opt.ckpt}!")

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (mol, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        if torch.cuda.is_available():
            mol = mol.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        output = model(mol).squeeze(1)

        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                      epoch,
                      idx + 1,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses))
            sys.stdout.flush()

    wandb.log({'train_loss': losses.avg}, step=epoch)


def validate(val_loader, model, opt):
    model.eval()

    losses = AverageMeter()
    if opt.loss == 'L1':
        criterion = torch.nn.L1Loss()
    elif opt.loss == 'MSE':
        criterion = torch.nn.MSELoss()
    elif opt.loss == 'huber':
        criterion = torch.nn.SmoothL1Loss()
    else:
        raise ValueError(f"Loss function {opt.loss} not supported!")

    rmse = AverageMeter()
    mae = AverageMeter()
    with torch.no_grad():
        for idx, (mol, labels) in enumerate(val_loader):
            mol = mol.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            output = model(mol).squeeze(1)

            loss = criterion(output, labels)
            losses.update(loss.item(), bsz)
            mae.update(abs(output - labels).mean().item(), bsz)
            rmse.update(((output - labels)**2).mean().item(), bsz)

    return losses.avg, math.sqrt(rmse.avg), mae.avg, rmse.avg


def main():
    seed_all(42)
    opt = parse_option()

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    save_file_best = os.path.join(opt.save_folder,
                                  f"{opt.model_name}_best.pth")
    save_file_last = os.path.join(opt.save_folder,
                                  f"{opt.model_name}_last.pth")
    best_error = 1e5

    start_epoch = 1
    if len(opt.resume):
        ckpt_state = torch.load(opt.resume)
        start_epoch = ckpt_state['epoch'] + 1
        best_error = ckpt_state['best_error']
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")

    # Start wandb run
    wandb.init(project="dl-project", config=opt)

    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, opt)

        valid_error, valid_rmse, valid_mae, valid_mse = validate(val_loader, model, opt)
        print('Val RMSE: {:.3f}'.format(valid_rmse))
        print('Val MAE: {:.3f}'.format(valid_mae))
        print('Val MSE: {:.3f}'.format(valid_mse))

        wandb.log({
            'valid_rmse': valid_rmse,
            'valid_mae': valid_mae,
            'valid_mse': valid_mse
        },
                  step=epoch)

        is_best = valid_error < best_error
        best_error = min(valid_error, best_error)
        print(f"Best Error: {best_error:.3f}")

        if is_best:
            torch.save(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_error': best_error
                }, save_file_best)

        torch.save(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'last_error': valid_error
            }, save_file_last)

    print("=" * 120)
    print("Test best model on test set...")
    checkpoint = torch.load(save_file_best)
    model.load_state_dict(checkpoint['state_dict'])
    print(
        f"Loaded best model, epoch {checkpoint['epoch']}, best val error {checkpoint['best_error']:.3f}"
    )
    test_loss, test_rmse, test_mae, test_mse = validate(test_loader, model, opt)
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Test MSE: {test_mse:.3f}")

    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
