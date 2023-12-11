from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset
from PIL import Image


class JagsawDataset(Dataset):
    def __init__(self, filename, transform):
        self.filename = filename
        self.labels = []
        self.image = []
        self.transform = transform
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f.read().splitlines():
                x = line[:line.find(',')]
                label = line[line.find(','):].split(',')[1]
                label = int(label)
                self.labels.append(label)
                self.image.append(x)
    
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        image = Image.open(self.image[idx]).convert('RGB')
        image = self.transform(image)
        label = self.labels
        return image, label[idx]
    


class ISIC2019Dataset(Dataset):
    def __init__(self, filenames, transform, label_transform):
        self.filenames = filenames
        self.labels = []
        self.image = []
        self.transform = transform
        self.label_transfrom = label_transform
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f.read().splitlines():
                    x = line[:line.find(',')]
                    label = line[line.find(','):].split(',')[1]
                    label = int(label)
                    self.labels.append(label)
                    self.image.append(x)
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        image = Image.open(self.image[idx]).convert('RGB')
        image = self.transform(image)
        label = self.label_transfrom(self.labels)
        return image, label[idx]



class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD([{'params': model.parameters()}],
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [300, 400], gamma=0.1)
    return optimizer, scheduler

def set_optimizer_cifar(opt, model):
    optimizer = optim.SGD([{'params': model.parameters()}],
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100,150,250,350], gamma=0.5)
    return optimizer, scheduler

def set_optimizer_one(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

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

def save__model(model, optimizer, epoch, save_file):
    print('==> Saving...')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


