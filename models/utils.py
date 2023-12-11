from models.wrn import WideResNet
from models.resnet import *
import torch
from torchvision.models import densenet121, mobilenet_v2, vgg11 
import numpy as np

def build_model(model_type, num_classes, device, args):
    if model_type == 'wrn':
        net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).to(device)
    elif model_type == 'res':
        net = Resnet_50(name='resnet50').to(device)
    elif model_type == 'dens':
        net = densenet121(pretrained=False, num_classes=num_classes).to(device)
    elif model_type == 'mobile':
        net = mobilenet_v2(pretrained=False, num_classes=num_classes).to(device)
    elif model_type == 'vgg':
        net = vgg11(pretrained=False, num_classes=num_classes).to(device)
    elif model_type == 'cider_resnet':
        if args:
            net = CIDERResNet(name=args.model_resnet, head=args.resnet_head).to(device)
        else:
            net = CIDERResNet(name='resnet34', head='mlp_cifar100').to(device)
    if args :
        if args.gpu is not None and len(args.gpu) > 1:
            gpu_list = [int(s) for s in args.gpu.split(',')]
            net = torch.nn.DataParallel(net, device_ids=gpu_list)
    return net

def build_test_model(model_type, num_classes, device):
    if model_type == 'res':
        net = Resnet_50(name='resnet50').to(device=device)
    return net