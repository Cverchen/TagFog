import random
import numpy as np
from PIL import Image
import torchvision.datasets as datasets
import torch
from torchvision import transforms
import os
# 读取 CIFAR-10 数据集中的一张图片
train_dataset = datasets.CIFAR10(root='./datasets/',transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor()]),download=True)
# 读取 CIFAR-100 数据集中的一张图片
# train_dataset = datasets.CIFAR100(root='./datasets/',transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor()]),download=True)
# 读取ImageNet-100 数据集中的一张图片
# train_dataset = datasets.ImageFolder(os.path.join('Datasets', 'ImageNet100', 'train'), transform=
                                    #  transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=6, pin_memory=False, sampler=None)
# for image,label in train_loader:
#     to_pil_image = transforms.ToPILImage()(image[0])
#     to_pil_image.save('test.png')
#     break
# img = Image.open('test.png')
# print(len(train_dataset))
def jigsaw_create(image):
    width, height = image.size
    patch_width = width // 4
    patch_height = height // 4
    patches = []
    for i in range(4):
        for j in range(4):
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

index = 0
for number in range(2):
    for image,label in train_loader:
        index += 1
        to_pil_image = transforms.ToPILImage()(image[0])
        jigsaw_image = jigsaw_create(to_pil_image)
        # jigsaw_image.save('datasets/ImageNet100/ImageNet100_general_jigsaw/'+str(index)+'.png')
        # print('第{}个jiasaw图片已经生成!'.format(index))
        jigsaw_image.save('datasets/cifar10_jigsaw/'+str(index)+'.png')
        print('第{}个jiasaw图片已经生成!'.format(index))
    
with open('datasets/cifar10_jigsaw/cifar10_jigsaw_100000.txt', 'w', encoding='utf-8') as f:
    for j in range(1, 100001):
        f.write('datasets/cifar10_jigsaw/'+str(j)+'.png,10\n')

# with open('datasets/ImageNet100/ImageNet100_general_jigsaw.txt', 'w', encoding='utf-8') as f:
#     for j in range(1, 128578):
#         f.write('datasets/ImageNet100/ImageNet100_general_jigsaw/'+str(j)+'.png,100\n')

# with open('datasets/cifar100_jigsaw/cifar100_jigsaw_100000.txt', 'w', encoding='utf-8') as f:
#     for j in range(1, 100001):
#         f.write('datasets/cifar100_jigsaw/'+str(j)+'.png,100\n')

# with open('datasets/cifar100_jigsaw/cifar100_jigsaw_5000.txt', 'w', encoding='utf-8') as f:
#     for j in range(1, 5001):
#         f.write('datasets/cifar100_jigsaw/'+str(j)+'.png,100\n')

# with open('datasets/cifar100_jigsaw/cifar100_jigsaw_200000.txt', 'w', encoding='utf-8') as f:
#     for j in range(1, 100001):
#         f.write('datasets/cifar100_jigsaw/'+str(j)+'.png,100\n')
#     for j in range(100001, 200001):
#         f.write('datasets/cifar100_jigsaw_more/'+str(j)+'.png,100\n')
