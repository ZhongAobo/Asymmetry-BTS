import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
from data.transforms import *
from data.datasets_aug import Brats_loadall_nii, Brats_loadall_test_nii, Brats_loadall_nii_aug
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from predict import AverageMeter, test_softmax


import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', '--batch_size', default=2, type=int, help='Batch size')
parser.add_argument('--datapath', default='/home/SHARE2/ZXY/RFNet/BRATS2020_Training_none_npy', type=str)
parser.add_argument('--dataname', default='BRATS2020', type=str)
parser.add_argument('--savepath', default='/home/SHARE2/ZXY/RF2Robust/results', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--mode', default='pretrain', type=str, help='pretrain, train or finetune')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--scale', default=8, type=int)


path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask  
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1ce', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']

def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    if args.dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print ('dataset is error')
        exit(0)
    model = models.Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model).cuda()

    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    if args.dataname in ['BRATS2020', 'BRATS2015']:
        train_file = 'train.txt'
        test_file = 'test.txt'
    elif args.dataname == 'BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = 'train3.txt'
        test_file = 'test3.txt'
    
    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)

    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Evaluate
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        test_score = AverageMeter()


    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = args.iter_per_epoch

    for epoch in range(args.num_epochs):
        train_iter = iter(train_loader)

        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]

            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True
            outputs = model(x, mask)

            ce_loss = criterions.softmax_weighted_loss(outputs['seg_pred'], target, num_cls=num_cls)
            dice_loss = criterions.dice_loss(outputs['seg_pred'], target, num_cls=num_cls)

            mse_loss = torch.nn.MSELoss()
            l2_loss =  torch.zeros(1).cuda().float()
            for name,params in model.module.named_parameters():
                # if not 'att_' in name and not 'fusion_' in name:
                #     l2_loss = l2_loss + 0.0001 * mse_loss(params,torch.zeros_like(params))
                if 'ce' in name:
                    l2_loss = l2_loss + 0.0001 * mse_loss(params,torch.zeros_like(params))
                if 'se' in name:
                    l2_loss = l2_loss + 0.0001 * mse_loss(params,torch.zeros_like(params))
                if 'image_de_' in name:
                    l2_loss = l2_loss + 0.0001 * mse_loss(params,torch.zeros_like(params))
                if 'mask_de' in name:
                    l2_loss = l2_loss + 0.0001 * mse_loss(params,torch.zeros_like(params))

            rec_loss =  torch.zeros(1).cuda().float()
            rec_loss = rec_loss + torch.mean(torch.abs(x[:,0:1,:,:,:] - outputs['reconstruct_flair']))
            rec_loss = rec_loss + torch.mean(torch.abs(x[:,1:2,:,:,:] - outputs['reconstruct_t1c']))
            rec_loss = rec_loss + torch.mean(torch.abs(x[:,2:3,:,:,:] - outputs['reconstruct_t1']))
            rec_loss = rec_loss + torch.mean(torch.abs(x[:,3:4,:,:,:] - outputs['reconstruct_t2']))

            kl_loss =  torch.zeros(1).cuda().float()
            kl_loss = kl_loss + criterions.kl_loss(outputs['mu_flair'], torch.log(torch.square(outputs['sigma_flair'])))
            kl_loss = kl_loss + criterions.kl_loss(outputs['mu_t1c'], torch.log(torch.square(outputs['sigma_t1c'])))
            kl_loss = kl_loss + criterions.kl_loss(outputs['mu_t1'], torch.log(torch.square(outputs['sigma_t1'])))
            kl_loss = kl_loss + criterions.kl_loss(outputs['mu_t2'], torch.log(torch.square(outputs['sigma_t2'])))

            loss  = ce_loss + dice_loss + l2_loss + 0.1*rec_loss + 0.1*kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('ce_loss', ce_loss.item(), global_step=step)
            writer.add_scalar('dice_loss', dice_loss.item(), global_step=step)
            writer.add_scalar('l2_loss', l2_loss.item(), global_step=step)
            writer.add_scalar('rec_loss', 0.1*rec_loss.item(), global_step=step)
            writer.add_scalar('kl_loss', 0.1*kl_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'ce_loss:{:.4f},'.format(ce_loss.item())
            msg += 'dice_loss:{:.4f},'.format(dice_loss.item())
            msg += 'l2_loss:{:.4f},'.format(l2_loss.item())
            msg += 'rec_loss:{:.4f}, kl_loss:{:.4f},'.format(0.1*rec_loss.item(), 0.1*kl_loss.item())
            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))

        ##########model save
       
        if (epoch+1) % 50 == 0 or (epoch>=args.num_epochs-5):
            file_name = os.path.join(ckpts, f'{args.mode}_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)
        
    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ##########Evaluate the last epoch model
    test_score = AverageMeter()
    with torch.no_grad():
       logging.info('###########test set wi/wo postprocess###########')
       for i, mask in enumerate(masks):
           logging.info('{}'.format(mask_name[i]))
           dice_score = test_softmax(
                           test_loader,
                           model,
                           dataname = args.dataname,
                           feature_mask = mask)
           test_score.update(dice_score)
       logging.info('Avg scores: {}'.format(test_score.avg))

if __name__ == '__main__':
    main()
