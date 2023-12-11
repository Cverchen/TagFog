import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm, pinv
from scipy.special import softmax
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.covariance import EmpiricalCovariance
from os.path import basename, splitext
from scipy.special import logsumexp
import sys
from os.path import dirname
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from models.utils import build_model, build_test_model
from datasets.uitls import build_dataset, build_cider_dataset, set_ood_loader_Imagenet
from datasets.cifar import CIFAR10, CIFAR100
import datasets.svhn_loader as svhn
import torchvision as tv
import pandas as pd
import torchvision.transforms as trn
import torchvision.datasets as dset
from models.resnet_big import SupConResNet,StandardResnet
from models.wrn import StandardWrn
from models.wrn import SupCEHeadWrn
import torch.backends.cudnn as cudnn
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import mmcv
from util_detect import *
import pickle
def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--model_resnet',type=str, default='resnet18',help='Supcon_resnet architecture')
    parser.add_argument('--dataset',type=str, default='cifar100',help='in_dataset')
    parser.add_argument('--save_path', type=str, default='feature/cifar100')
    parser.add_argument('--fc_save_path', type=str, default='fc/cifar100/fc.pkl')
    parser.add_argument('--end_path', type=str, default='')
    parser.add_argument('--n_classes', type=int, default=100)
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    return args

def set_model(args):
    model = SupConResNet(name=args.model_resnet, dataset=args.dataset)
    # model = StandardResnet(name=args.model_resnet, dataset=args.dataset)
    # model = StandardWrn(num_class=args.n_classes)
    # model = SupCEHeadWrn(num_class=args.n_classes+1)
    # ckpt = torch.load('saved_model/'+args.dataset+args.end_path+'/cifar100_200epoch_resnet18_lr_0.05_150update.pth', map_location='cpu')
    ckpt = torch.load('saved_model/'+args.dataset+args.end_path+'/last.pth', map_location='cpu')
    state_dict = ckpt['model']
    # state_dict = ckpt
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:  
            model.encoder = torch.nn.DataParallel(model.encoder)
            model.classifier = torch.nn.DataParallel(model.classifier)
            model.addmlp = torch.nn.DataParallel(model.addmlp)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict)
    return model

def extract_feature_id(args, model, batch_size):
    # ID Dataset
    train_data, num_classes = build_dataset(args.dataset, 'train', eval=True)
    test_data, num_classes = build_cider_dataset(args.dataset, 'test')
    train_loader = torch.utils.data.DataLoader(
        train_data,batch_size=batch_size,shuffle=False,
        num_workers=4,pin_memory=True,drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,batch_size=batch_size,shuffle=False,
        num_workers=4,pin_memory=True,drop_last=False
    )
    id_train_features = []
    id_test_features = []
    for i in range(num_classes):
        id_train_features.append([])
        id_test_features.append([])
    # features 保存了10类的output特征 features[i]表示第i类的所有样本的output feature
    with torch.no_grad():
        print('Dataset:{} sum:{}'.format(args.dataset,len(train_data)))
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            for image, label, idx in tqdm(train_loader):
                image = image.cuda()
                feat_batch = model.encoder(image)
                # if maha is True:
                    # feat_batch = model.addmlp(feat_batch)
                feat_batch = feat_batch.cpu().numpy()
                for i in range(feat_batch.shape[0]):
                    id_train_features[label[i]].append(feat_batch[i])
            print('Dataset:{} sum:{}'.format(args.dataset,len(test_data)))
            with torch.no_grad():
                for image, label, idx in tqdm(test_loader):
                    image = image.cuda()
                    feat_batch = model.encoder(image)
                    feat_batch = feat_batch.cpu().numpy()
                    for i in range(feat_batch.shape[0]):
                        id_test_features[label[i]].append(feat_batch[i]) 
        else:
            for image, label in tqdm(train_loader):
                image = image.cuda()
                feat_batch = model.encoder(image)
                feat_batch = feat_batch.cpu().numpy()
                for i in range(feat_batch.shape[0]):
                    id_train_features[label[i]].append(feat_batch[i])
            print('Dataset:{} sum:{}'.format(args.dataset,len(test_data)))
            for image, label in tqdm(test_loader):
                image = image.cuda()
                feat_batch = model.encoder(image)
                feat_batch = feat_batch.cpu().numpy()
                for i in range(feat_batch.shape[0]):
                    id_test_features[label[i]].append(feat_batch[i])

    # all_id_train_features = np.concatenate(id_train_features, axis=0) # [traind_data's length, 128]
    all_id_train_features = np.array(id_train_features)
    all_id_test_features = np.concatenate(id_test_features, axis=0) # [tested_data's length, 128]
    np.save(os.path.join(args.save_path,args.dataset+'_train.npy'), all_id_train_features)
    np.save(os.path.join(args.save_path,args.dataset+'_test.npy'), all_id_test_features)

def extract_feature_ood(args, OOD_data_list, batch_size):
    feature_oods = []
    for i in range(len(OOD_data_list)):
        feature_oods.append([])
    index = 0
    for data_name in OOD_data_list:       
        ood_data, _ = build_cider_dataset(data_name, mode="test")
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size, shuffle=True,
                                                num_workers=4, pin_memory=True, drop_last = False)
        print('Dataset: {}  Sum:{}'.format(OOD_data_list[index], len(ood_data)))
        with torch.no_grad():
            if data_name == 'cifar10' or data_name == 'cifar100':
                for image, labels, idx in tqdm(ood_loader):
                    image = image.cuda()
                    feat_batch = model.encoder(image)# 得到输出特征
                    feat_batch = feat_batch.cpu().numpy()
                    for i in range(feat_batch.shape[0]):
                        feature_oods[index].append(feat_batch[i])
                np.save(os.path.join(args.save_path, data_name+'.npy'), feature_oods[index])
            else:
                for image, labels in tqdm(ood_loader):
                    image = image.cuda()
                    feat_batch = model.encoder(image)# 得到输出特征
                    feat_batch = feat_batch.cpu().numpy()
                    for i in range(feat_batch.shape[0]):
                        feature_oods[index].append(feat_batch[i])
                np.save(os.path.join(args.save_path, data_name+'.npy'), feature_oods[index])
        index += 1




def extract_fc(args, model):
    mmcv.mkdir_or_exist(dirname(args.fc_save_path))
    w = model.classifier.weight.cpu().detach().numpy()
    b = model.classifier.bias.cpu().detach().numpy()
    with open(args.fc_save_path, 'wb') as f:
        pickle.dump([w,b],f)
    print('fc weight has extracted!')


if __name__ == '__main__':
    args = parse_option()
    model = set_model(args)
    batch_size = 256
    model.eval()
    OOD_data_list = ["Textures", "SVHN", "iSUN", "LSUN-C", "LSUN-R", "Places365"]
    maha = False    # 是否需要马氏距离测试
    extract = True
    if maha is False:
        extract_fc(args, model)
    if extract is True:
        extract_feature_id(args, model, batch_size=batch_size)
        print('id features has extracted!')
        extract_feature_ood(args, OOD_data_list, batch_size=batch_size)
        print('OOD features has extracted!')    
    