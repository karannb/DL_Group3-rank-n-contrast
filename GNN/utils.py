import math
import torch
import random

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    eta_min = lr * (args.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def set_optimizer(opt, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate,
                                 weight_decay=opt.weight_decay) #momentum=opt.momentum,

    return optimizer

def seed_all(seed: int):
    """Setup random state from a seed for `torch`, `random` and optionally `numpy` (if can be imported).

    Args:
        seed: Random state seed
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass