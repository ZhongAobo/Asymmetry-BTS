from re import sub
import torch.nn.functional as F
import torch
import logging
import torch.nn as nn
from itertools import chain, combinations

__all__ = ['sigmoid_dice_loss','softmax_dice_loss','GeneralizedDiceLoss','FocalLoss', 'dice_loss']

cross_entropy = F.cross_entropy

def dice_loss(output, target, num_cls=5, eps=1e-7):
    target = target.float()
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:])
        l = torch.sum(output[:,i,:,:,:])
        r = torch.sum(target[:,i,:,:,:])
        if i == 0:
            dice = 2.0 * num / (l+r+eps)
        else:
            dice += 2.0 * num / (l+r+eps)
    return 1.0 - 1.0 * dice / num_cls#计算dice_loss

def softmax_weighted_loss(output, target, num_cls=5):
    target = target.float()
    B, _, H, W, Z = output.size()#batchsize 
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss
            
def softmax_loss(output, target, num_cls=5):
    target = target.float()
    _, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        if i == 0:
            cross_loss = -1.0 * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss

def FocalLoss(output, target, alpha=0.25, gamma=2.0):
    target[target == 4] = 3 # label [4] -> [3]
    # target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4,H,W,D]
    if output.dim() > 2:
        output = output.view(output.size(0), output.size(1), -1)  # N,C,H,W,D => N,C,H*W*D
        output = output.transpose(1, 2)  # N,C,H*W*D => N,H*W*D,C
        output = output.contiguous().view(-1, output.size(2))  # N,H*W*D,C => N*H*W*D,C
    if target.dim() == 5:
        target = target.contiguous().view(target.size(0), target.size(1), -1)
        target = target.transpose(1, 2)
        target = target.contiguous().view(-1, target.size(2))
    if target.dim() == 4:
        target = target.view(-1) # N*H*W*D
    # compute the negative likelyhood
    logpt = -F.cross_entropy(output, target)
    pt = torch.exp(logpt)
    # compute the loss
    loss = -((1 - pt) ** gamma) * logpt
    # return loss.sum()
    return loss.mean()

def dice(output, target,eps =1e-5): # soft dice loss
    target = target.float()
    # num = 2*(output*target).sum() + eps
    num = 2*(output*target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def sigmoid_dice_loss(output, target,alpha=1e-5):
    # output: [-1,3,H,W,T]
    # target: [-1,H,W,T] noted that it includes 0,1,2,4 here
    loss1 = dice(output[:,0,...],(target==1).float(),eps=alpha)
    loss2 = dice(output[:,1,...],(target==2).float(),eps=alpha)
    loss3 = dice(output[:,2,...],(target == 4).float(),eps=alpha)
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))
    return loss1+loss2+loss3


def softmax_dice_loss(output, target,eps=1e-5): #
    # output : [bsize,c,H,W,D]
    # target : [bsize,H,W,D]
    loss1 = dice(output[:,1,...],(target==1).float())
    loss2 = dice(output[:,2,...],(target==2).float())
    loss3 = dice(output[:,3,...],(target==4).float())
    logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(1-loss1.data, 1-loss2.data, 1-loss3.data))

    return loss1+loss2+loss3


# Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
def GeneralizedDiceLoss(output,target,eps=1e-5,weight_type='square'): # Generalized dice loss
    """
        Generalised Dice : 'Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations'
    """

    # target = target.float()
    if target.dim() == 4:
        target[target == 4] = 3 # label [4] -> [3]
        target = expand_target(target, n_class=output.size()[1]) # [N,H,W,D] -> [N,4，H,W,D]

    output = flatten(output)[1:,...] # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:,...] # [class, N*H*W*D]

    target_sum = target.sum(-1) # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :',weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)
    loss2 = 2*intersect[1] / (denominator[1] + eps)
    loss3 = 2*intersect[2] / (denominator[2] + eps)
    #logging.info('1:{:.4f} | 2:{:.4f} | 4:{:.4f}'.format(loss1.data, loss2.data, loss3.data))

    return 1 - 2. * intersect_sum / denominator_sum, [loss1.data, loss2.data, loss3.data]


def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:,1,:,:,:] = (x == 1)
        xx[:,2,:,:,:] = (x == 2)
        xx[:,3,:,:,:] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:,0,:,:,:] = (x == 1)
        xx[:,1,:,:,:] = (x == 2)
        xx[:,2,:,:,:] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)


MODALITIES_img = ['Flair', 'T1c', 'T1', 'T2']

def all_subsets(l):
    #Does not include the empty set and l
    return list(chain(*map(lambda x: combinations(l, x), range(1, len(l)+1))))

SUBSETS_MODALITIES = all_subsets(MODALITIES_img)
def compute_KLD(means,logvars, choices):

    mu_prior = torch.zeros_like(means[MODALITIES_img[0]])   #[B,4,80,80,80]
    log_prior = torch.zeros_like(means[MODALITIES_img[0]])  #[B,4,80,80,80]

    full_means,full_logvars = Product_Gaussian_main(means, logvars, MODALITIES_img, choices)
    sum_inter_KLD = 0
    sum_prior_KLD = 0
    for subset in SUBSETS_MODALITIES:
        sub_means,sub_logvars = Product_Gaussian(means,logvars,subset)
        sum_inter_KLD = sum_inter_KLD + KL_divergence(full_means,full_logvars,mu_prior,log_prior)
        sum_prior_KLD = sum_prior_KLD + KL_divergence(sub_means,sub_logvars,mu_prior,log_prior)
    return 1/14*sum_inter_KLD, 1/15*sum_prior_KLD

def Product_Gaussian_main(means, logvars, list_mod, choices):
    mu_prior = torch.zeros_like(means[MODALITIES_img[0]])
    log_prior = torch.zeros_like(means[MODALITIES_img[0]])
    eps = 1e-5
    a = [1/(torch.exp(logvars[mod]) + eps)  for mod in list_mod]
    b = [means[mod]/(torch.exp(logvars[mod]) + eps) for mod in list_mod]
    T = torch.zeros(4,mu_prior.shape[0],mu_prior.shape[1],mu_prior.shape[2],mu_prior.shape[3],mu_prior.shape[4]).cuda()
    mu = torch.zeros(4,log_prior.shape[0],log_prior.shape[1],log_prior.shape[2],log_prior.shape[3],log_prior.shape[4]).cuda()
    for i in range(4):
        if choices[i]:
            T[i,...] = a[i]
            mu[i,...] = b[i]

    T = torch.cat((T,(1+log_prior).unsqueeze(0)),0)
    mu = torch.cat((mu,(mu_prior).unsqueeze(0)),0)

    posterior_means = torch.sum(mu,0) / torch.sum(T,0)
    var = 1 / torch.sum(T,0)
    posterior_logvars = torch.log(var+eps)
    return posterior_means,posterior_logvars

def Product_Gaussian(means, logvars, list_mod): 
    mu_prior = torch.zeros_like(means[MODALITIES_img[0]])
    log_prior = torch.zeros_like(means[MODALITIES_img[0]])
    eps = 1e-2
    T = [1/(torch.exp(logvars[mod]) + eps) for mod in list_mod] + [1+log_prior]
    mu = [means[mod]/(torch.exp(logvars[mod]) + eps) for mod in list_mod] + [mu_prior]

    a = T[0]
    posterior_means = torch.zeros(len(T),a.shape[0],a.shape[1],a.shape[2],a.shape[3],a.shape[4]).cuda()
    var = torch.zeros(len(T),a.shape[0],a.shape[1],a.shape[2],a.shape[3],a.shape[4]).cuda()
    for i in range(len(T)):
        posterior_means[i,...] = mu[i] / T[i]            ######################
        var[i,...] = 1 / T[i]                         ######################
    posterior_logvars = torch.log(var+eps)
    return posterior_means,posterior_logvars

def KL_divergence(mu_1, logvar_1, mu_2, logvar_2):
    var_1 = torch.exp(logvar_1)
    var_2 = torch.exp(logvar_2)
    return 1/2*torch.mean(-1 + logvar_2 - logvar_1 + (var_1+torch.square(mu_1-mu_2))/(var_2+1e-2))
