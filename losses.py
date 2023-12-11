"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CELoss(nn.Module):
    '''label smooth cross entropy loss'''
    def __init__(self, label_smooth=None, class_num=10):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num
    def forward(self, pred, target):
        '''
        Args:
            preds: prediction of model output [N,M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1) # softmax+log
            target = F.one_hot(target, self.class_num)
            target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1),max=1.0-self.label_smooth)
            # target [512, 10]  logprobs [512, 128]
            loss = -1*torch.sum(target*logprobs, 1)
        else:
            # standard cross entropy loss
            loss = -1.*pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))
        return loss.mean() 


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class Comloss(nn.Module):
    def __init__(self, text_prototype):
        super(Comloss, self).__init__()
        self.text_prototype = text_prototype
    def forward(self, im_k):#(im_k:(100000, 768))
        comloss = 0
        # (100000, 768) * (768, 10)  ---> (100000, 10)
        comloss += torch.logsumexp(torch.logsumexp(torch.matmul(im_k, self.text_prototype.T), dim=1), dim=-1)
        comloss /= im_k.shape[0]
        return comloss
        

class Itloss(nn.Module):
    def __init__(self, opt, num_class, alpha=1, beta=1):
        super(Itloss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.beta = beta
        self.opt = opt
    def forward(self, features_front, logits, labels, text_prototype, mask=None):
        loss_ce = F.cross_entropy(logits, labels)

         # clip text_encoder prototype
        prototypes = text_prototype
        prototypes = F.normalize(prototypes, dim=1)
        
        proxy_labels = torch.arange(0, self.num_class).cuda()
        Labels = labels.contiguous().view(-1, 1)
        
        mask = torch.eq(Labels, proxy_labels.T).float().cuda() #bz, cls
        # compute logits
        temperature = 0.1
        feat_dot_prototype = torch.div(
            torch.matmul(features_front, prototypes.T),
            temperature)
        # for numerical stability
        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) 
        # loss
        simloss = -1*mean_log_prob_pos.mean()

        # # text loss
        # prototypes = text_prototype
        # num_cls = self.num_class
        # labels = torch.arange(0, num_cls).cuda()
        # labels = labels.contiguous().view(-1, 1)
        # labels = labels.contiguous().view(-1, 1)

        # mask = (1- torch.eq(labels, labels.T).float()).cuda()
        # logits = torch.div(
        #     torch.matmul(prototypes, prototypes.T),
        #     temperature)
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(num_cls).view(-1, 1).cuda(),
        #     0
        # )
        # mask = mask * logits_mask
        # mean_prob_neg = torch.log((mask * torch.exp(logits)).sum(1) / mask.sum(1))
        # mean_prob_neg = mean_prob_neg[~torch.isnan(mean_prob_neg)]
        # textloss = mean_prob_neg.mean()

        c = self.num_class
        textloss = 0

        for i in range(c):
            for j in range(c):
                if i != j:
                    textloss += torch.exp(torch.matmul(prototypes[i], prototypes[j]))
        textloss = torch.log(textloss)
        textloss /= c

        loss = self.alpha * simloss + self.beta * loss_ce + textloss

        return loss


class Itloss_jiasaw(nn.Module):
    def __init__(self, opt, num_class, alpha=1, beta=1):
        super(Itloss_jiasaw, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.beta = beta
        self.opt = opt
    def forward(self, features_front, logits, labels, text_prototype, mask=None):
        loss_ce = F.cross_entropy(logits, labels)

         # clip text_encoder prototype
        prototypes = text_prototype
        prototypes = F.normalize(prototypes, dim=1)
        
        proxy_labels = torch.arange(0, self.num_class-1).cuda()
        Labels = labels.contiguous().view(-1, 1)
        
        mask = torch.eq(Labels, proxy_labels.T).float().cuda() #bz, cls
        # compute logits
        temperature = 0.1
        feat_dot_prototype = torch.div(
            torch.matmul(features_front, prototypes.T),
            temperature)
        # for numerical stability
        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) 
        # loss
        simloss = -1*mean_log_prob_pos.mean()

        # jiasaw样本与所有原型尽量远离
        seploss = 0
        num = 0
        if self.opt.jigsaw == 'True':
            for i in range(len(labels)):
                if labels[i] == self.num_class:
                    seploss += torch.logsumexp(torch.matmul(features_front[i], prototypes.T).unsqueeze(0), dim=1).squeeze(0)
                    num += 1
        if num != 0:            
            seploss /= num


        c = self.num_class - 1
        textloss = 0

        for i in range(c):
            for j in range(c):
                if i != j:
                    textloss += torch.exp(torch.matmul(prototypes[i], prototypes[j]))
        textloss = torch.log(textloss)
        textloss /= c

        loss = self.alpha * simloss + self.beta * loss_ce + textloss + seploss

        return loss

        
class Ourloss(nn.Module):
    def __init__(self, opt, num_class, alpha=1, beta=1, text_prototype=None):
        super(Ourloss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.beta = beta
        self.text_prototype = text_prototype
        self.opt = opt
    def forward(self, features_front, logits, labels, mask=None): # features_front [bsz+bsz, feat_dim]
        loss_ce = F.cross_entropy(logits, labels)

         # clip text_encoder prototype
        prototypes = self.text_prototype # (num_classes, 768)
        prototypes = F.normalize(prototypes, dim=1)
        
        proxy_labels = torch.arange(0, self.num_class-1).cuda()
        Labels = labels.contiguous().view(-1, 1)
        
        mask = torch.eq(Labels, proxy_labels.T).float().cuda() #bz, cls
        # compute logits
        # temperature = 0.1
        temperature = self.opt.temp_ci
        feat_dot_prototype = torch.div(
            torch.matmul(features_front, prototypes.T),
            temperature)
        # for numerical stability
        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) 
        # loss
        simloss = -1*mean_log_prob_pos.mean()

        if self.opt.jigsaw == 'True':
            loss = self.alpha * simloss +  loss_ce
        else:
            loss = self.alpha * simloss + self.beta * loss_ce

        return loss
    
class OurlossonlyCE(nn.Module):
    def __init__(self, opt, num_class, alpha=1, beta=1, text_prototype=None):
        super(OurlossonlyCE, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.beta = beta
        self.text_prototype = text_prototype
        self.opt = opt
    def forward(self, features_front, logits, labels, mask=None): # features_front [bsz+bsz, feat_dim]
        loss_ce = F.cross_entropy(logits, labels)

        
        loss = loss_ce

        return loss


class AblationLoss(nn.Module):
    def __init__(self, opt, num_class, alpha=1, beta=1, text_prototype=None):
        super(AblationLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.beta = beta
        self.text_prototype = text_prototype
        self.opt = opt
    def forward(self, features_front, logits, labels,mask=None): # features_front [bsz+bsz, feat_dim]
        loss_ce = F.cross_entropy(logits, labels)

        
         # clip text_encoder prototype
        prototypes = self.text_prototype # (num_classes, 768)
        prototypes = F.normalize(prototypes, dim=1)
        
        proxy_labels = torch.arange(0, self.num_class).cuda()
        Labels = labels.contiguous().view(-1, 1)
        
        mask = torch.eq(Labels, proxy_labels.T).float().cuda() #bz, cls
        # compute logits
        temperature = 0.1
        feat_dot_prototype = torch.div(
            torch.matmul(features_front, prototypes.T),
            temperature)
        # for numerical stability
        logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
        logits = feat_dot_prototype - logits_max.detach()
        # compute log_prob
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) 
        # loss
        simloss = -1*mean_log_prob_pos.mean()

        seploss = 0
        num = 0
        if self.opt.jigsaw == 'True':
            for i in range(len(labels)):
                if labels[i] == self.num_class:
                    seploss += torch.logsumexp(torch.matmul(features_front[i], prototypes.T).unsqueeze(0), dim=1).squeeze(0)
                    num += 1
        if num != 0:            
            seploss /= num

        if self.opt.jigsaw == 'True':
            loss = self.alpha * simloss + self.beta * loss_ce + seploss 
        else:
            loss = self.alpha * simloss + self.beta * loss_ce

        return loss

        
