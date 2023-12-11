import random
import numpy as np
from PIL import Image
import torchvision.datasets as datasets
import torch
from torchvision import transforms
import os
import argparse


def parse_option():
    parser = argparse.ArgumentParser('create jiasaw data for training')
    parser.add_argument('--dataset', type=str,default='cifar10',
                        help='cifar10, cifar100, ImageNet100-I, ImageNet100-II')
    parser.add_argument('--jig_per_img', type=int, default=1, help='number of jiasaw for every training image')
    parser.add_argument('--num_patch', type=int, default=4, help='a image into num_patch x num_patch patches')
    opt = parser.parse_args()
    return opt

def load_data(opt):
    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./datasets/',transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor()]),download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./datasets/',transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor()]),download=True)
    elif opt.dataset == 'ImageNet100-I':
        train_dataset = datasets.ImageFolder(os.path.join('datasets', 'ImageNet100-I', 'train'), transform=
                                     transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()]))
    elif opt.dataset == 'ImageNet100-II':
        train_dataset = datasets.ImageFolder(os.path.join('datasets', 'ImageNet100-II', 'train'), transform=
                                     transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False,
            num_workers=6, pin_memory=False, sampler=None)
    return train_loader
# for image,label in train_loader:
#     to_pil_image = transforms.ToPILImage()(image[0])
#     to_pil_image.save('test.png')
#     break
# img = Image.open('test.png')
# print(len(train_dataset))
def jigsaw_create(image, num_patch=4):
    width, height = image.size
    patch_width = width // num_patch
    patch_height = height // num_patch
    patches = []
    for i in range(num_patch):
        for j in range(num_patch):
            x1 = i * patch_width
            y1 = j * patch_height
            x2 = x1 + patch_width
            y2 = y1 + patch_height
            patch = image.crop((x1, y1, x2, y2))
            patches.append(patch)
    # 随机打乱九个块
    random.shuffle(patches)
    # 重新组合块，生成新的图片
    new_img = Image.new('RGB', (width, height))
    x = 0
    y = 0
    for patch in patches:
        new_img.paste(patch, (x, y))
        x += patch_width
        if x >= width:
            x = 0
            y += patch_height
    return new_img
# # 显示原始图片和新的图片
# # to_pil_image.save('test.png')
# # new_img = jigsaw_create(to_pil_image)
# # new_img.save('random_test.png')
def create_data(opt, train_loader):
    if opt.dataset == 'cifar10':
        folder = 'datasets/cifar10_jigsaw'
        if not os.path.exists(folder):
            os.makedirs(folder)
        index = 0
        for _ in range(opt.jig_per_img):
            for image,label in train_loader:
                index += 1
                to_pil_image = transforms.ToPILImage()(image[0])
                jigsaw_image = jigsaw_create(to_pil_image, opt.num_patch)
                # jigsaw_image.save('datasets/ImageNet100/ImageNet100_general_jigsaw/'+str(index)+'.png')
                # print('第{}个jiasaw图片已经生成!'.format(index))
                jigsaw_image.save('datasets/cifar10_jigsaw/'+str(index)+'.png')
                print('第{}个jiasaw图片已经生成!'.format(index))
        with open('datasets/cifar10_jigsaw/cifar10_jigsaw_{}.txt'.format(index), 'w', encoding='utf-8') as f:
            for j in range(1, index+1):
                f.write('datasets/cifar10_jigsaw/'+str(j)+'.png,10\n')
    elif opt.dataset == 'cifar100':
        folder = 'datasets/cifar100_jigsaw'
        if not os.path.exists(folder):
            os.makedirs(folder)
        index = 0
        for _ in range(opt.jig_per_img):
            for image,label in train_loader:
                index += 1
                to_pil_image = transforms.ToPILImage()(image[0])
                jigsaw_image = jigsaw_create(to_pil_image, opt.num_patch)
                # jigsaw_image.save('datasets/ImageNet100/ImageNet100_general_jigsaw/'+str(index)+'.png')
                # print('第{}个jiasaw图片已经生成!'.format(index))
                jigsaw_image.save('datasets/cifar100_jigsaw/'+str(index)+'.png')
                print('第{}个jiasaw图片已经生成!'.format(index))
        with open('datasets/cifar100_jigsaw/cifar100_jigsaw.txt', 'w', encoding='utf-8') as f:
            for j in range(1, index+1):
                f.write('datasets/cifar100_jigsaw/'+str(j)+'.png,100\n')
    elif opt.dataset == 'ImageNet100-I':
        folder = 'datasets/ImageNet100-I_jigsaw'
        if not os.path.exists(folder):
            os.makedirs(folder)
        index = 0
        for _ in range(opt.jig_per_img):
            for image,label in train_loader:
                index += 1
                to_pil_image = transforms.ToPILImage()(image[0])
                jigsaw_image = jigsaw_create(to_pil_image, opt.num_patch)
                jigsaw_image.save('datasets/ImageNet100-I_jigsaw/'+str(index)+'.png')
                print('第{}个jiasaw图片已经生成!'.format(index))
        with open('datasets/ImageNet100-I_jigsaw/ImageNet100-I_jigsaw.txt', 'w', encoding='utf-8') as f:
            for j in range(1, index+1):
                f.write('datasets/ImageNet100-I_jigsaw/'+str(j)+'.png,100\n')
    elif opt.dataset == 'ImageNet100-II':
        folder = 'datasets/ImageNet100-II_jigsaw'
        if not os.path.exists(folder):
            os.makedirs(folder)
        index = 0
        for _ in range(opt.jig_per_img):
            for image,label in train_loader:
                index += 1
                to_pil_image = transforms.ToPILImage()(image[0])
                jigsaw_image = jigsaw_create(to_pil_image, opt.num_patch)
                jigsaw_image.save('datasets/ImageNet100-II_jigsaw/'+str(index)+'.png')
                print('第{}个jiasaw图片已经生成!'.format(index))
        with open('datasets/ImageNet100-II_jigsaw/ImageNet100-II_jigsaw.txt', 'w', encoding='utf-8') as f:
            for j in range(1, index+1):
                f.write('datasets/ImageNet100-II_jigsaw/'+str(j)+'.png,100\n')

def main():
    opt = parse_option()
    train_loader = load_data(opt)
    create_data(opt, train_loader)

if __name__ == '__main__':
    main()