import argparse
import warnings
import os
import sys
import logging
import torch
import time
import math
from model import GCNMLP, ModelArgs
from torch_geometric.loader import DataLoader
from dataset import *
from utils import *
import wandb
import yaml

print = logging.info
# These warnings are unnecessarily verbose
warnings.filterwarnings('ignore',
                        category=UserWarning,
                        message='TypedStorage is deprecated')


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # Logging and checkpoint options
    parser.add_argument('--print_freq',
                        type=int,
                        default=2,
                        help='print frequency')
    parser.add_argument('--save_freq',
                        type=int,
                        default=50,
                        help='save frequency')
    parser.add_argument('--save_curr_freq',
                        type=int,
                        default=1,
                        help='save curr last frequency')

    # Optimization options
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
                        default=1e-2,
                        help='learning rate')
    parser.add_argument('--lr_decay_rate',
                        type=float,
                        default=0.0,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-6,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--trial',
                        type=str,
                        default='0',
                        help='id for recording multiple runs')

    # Model options
    parser.add_argument('--in_channels',
                        type=int,
                        default=9,
                        help='input channels')
    parser.add_argument('--out_channels',
                        type=int,
                        default=100,
                        help='output channels')
    parser.add_argument('--hidden_channels',
                        type=int,
                        default=64,
                        help='hidden channels')
    parser.add_argument('--num_layers',
                        type=int,
                        default=5,
                        help='number of layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help='dropout rate')

    # Other options
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

    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_path = './save/{}_models'.format(opt.dataset)
    opt.model_name = 'L1_{}_GNN_ep_{}_lr_{}_d_{}_wd_{}_mmt_{}_bsz_{}_aug_{}_trial_{}'. \
        format(opt.dataset, opt.epochs, opt.learning_rate, opt.lr_decay_rate, opt.weight_decay, opt.momentum,
               opt.batch_size, opt.aug, opt.trial)
    if len(opt.resume):
        opt.model_name = opt.resume.split('/')[-2]

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    else:
        print('WARNING: folder already exists.')
        
    opt.experiment_name = "L1-GNNMLP"

    # Create a logging object
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s",
                        handlers=[
                            logging.FileHandler(
                                os.path.join(opt.save_folder, 'training.log')),
                            logging.StreamHandler()
                        ])

    print(f"Model name: {opt.model_name}")
    print(f"Options: {opt}")

    return opt


def set_loader(opt):
    '''
    Load the dataset and return data loaders
    '''
    train_dataset = globals()[opt.dataset](data_folder=opt.data_folder,
                                           split='train')
    val_dataset = globals()[opt.dataset](data_folder=opt.data_folder,
                                         split='valid')
    test_dataset = globals()[opt.dataset](data_folder=opt.data_folder,
                                          split='test')

    print(f'Train set size: {train_dataset.__len__()}\t'
          f'Val set size: {val_dataset.__len__()}\t'
          f'Test set size: {test_dataset.__len__()}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
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


def set_model_and_loss(opt):
    '''
    Load the model, loss function and move to GPU
    '''
    model_args = ModelArgs()
    model_args.in_channels = opt.in_channels
    model_args.out_channels = opt.out_channels
    model_args.hidden_channels = opt.hidden_channels
    model_args.num_layers = opt.num_layers
    model_args.dropout = opt.dropout
    model = GCNMLP(model_args)
    criterion = torch.nn.L1Loss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        torch.backends.cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    '''
    One epoch trainer function
    '''
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (mol, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = labels.shape[0]

        if torch.cuda.is_available():
            mol = mol.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        output = model(mol).squeeze(1)

        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            to_print = 'Train: [{0}][{1}/{2}]\t'\
                       'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                       'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                       'loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses
            )
            print(to_print)
            sys.stdout.flush()
    wandb.log({
        'train_loss': losses.avg,
    }, step=epoch)


def validate(val_loader, model):
    '''
    Calculate the validation loss
    '''
    model.eval()

    losses = AverageMeter()
    rmse = AverageMeter()
    mae = AverageMeter()
    criterion_l1 = torch.nn.L1Loss()

    with torch.no_grad():
        for _, (mol, labels) in enumerate(val_loader):
            mol = mol.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            output = model(mol).squeeze(1)

            loss_l1 = criterion_l1(output, labels)
            losses.update(loss_l1.item(), bsz)
            mae.update(abs(output - labels).mean().item(), bsz)
            rmse.update(((output - labels)**2).mean().item(), bsz)

    return losses.avg, math.sqrt(rmse.avg), mae.avg, rmse.avg


def main():
    seed_all(42)
    opt = parse_option()

    # Setup wandb
    wandb.init(project='dl-project', config=opt)

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model_and_loss(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    start_epoch = 1
    if len(opt.resume):
        ckpt_state = torch.load(opt.resume)
        model.load_state_dict(ckpt_state['model'])
        optimizer.load_state_dict(ckpt_state['optimizer'])
        start_epoch = ckpt_state['epoch'] + 1
        print(f"<=== Epoch [{ckpt_state['epoch']}] Resumed from {opt.resume}!")

    best_error = 1e5
    save_file_best = os.path.join(opt.save_folder, 'best.pth')

    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, opt)

        valid_error, valid_rmse, valid_mae, valid_mse = validate(val_loader, model)
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

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        if epoch % opt.save_curr_freq == 0:
            save_file = os.path.join(opt.save_folder,
                                     'curr_last.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

        if is_best:
            torch.save(
                {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_error': best_error
                }, save_file_best)

    print("=" * 120)
    print("Test best model on test set...")
    checkpoint = torch.load(save_file_best)
    model.load_state_dict(checkpoint['model'])
    print(
        f"Loaded best model, epoch {checkpoint['epoch']}, best val error {checkpoint['best_error']:.3f}"
    )
    test_loss, test_rmse, test_mae, test_mse = validate(test_loader, model)
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Test MSE: {test_mse:.3f}")

    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
