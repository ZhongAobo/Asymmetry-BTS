#coding=utf-8
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
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from predict import AverageMeter, test_softmax

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default=r'/home/SHARE2/ZXY/RFNet/BRATS2020_Training_none_npy', type=str)
parser.add_argument('--dataname', default='BRATS2020', type=str)
parser.add_argument('--savepath', default='results', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--mode', default='finetune', type=str, help='pretrain, train or finetune')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=1, type=int)
parser.add_argument('--seed', default=1212, type=int)
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
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print ('dataset is error')
        exit(0)
    model = models.Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model).cuda()

    model_teacher = models.Model(num_cls=num_cls)
    model_teacher = torch.nn.DataParallel(model_teacher).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)#lr*(1-epoch/num_epochs)^0.9
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ##########Setting data
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

    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        model_teacher.load_state_dict(checkpoint['state_dict'])

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = args.iter_per_epoch
    train_iter = iter(train_loader)
    for epoch in range(args.num_epochs):
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

            mask_teacher = torch.zeros_like(mask)

            for m in range(mask.shape[0]):
                modal_nums = sum(mask[m]).item()
                if modal_nums == 1:
                    if mask[m,0] == True:
                        two_modals = [[True, False, True, False], [True, False, False, True], [True, True, False, False]]
                    elif mask[m,1] == True:
                        two_modals = [[False, True, False, True], [False, True, True, False], [True, True, False, False]]
                    elif mask[m,2] == True:
                        two_modals = [[False, True, True, False], [True, False, True, False], [False, False, True, True]]
                    elif mask[m,3] == True:
                        two_modals = [[False, True, False, True], [False, False, True, True], [True, False, False, True]]
                    two_modals = torch.from_numpy(np.array(two_modals))   
                    choice = np.random.choice(3, 1)
                    mask_teacher[m:m+1,:] = two_modals[choice]
                elif modal_nums == 2:
                    if mask[m,0] == True and mask[m,1] == True:
                        three_modals = [[True, True, True, False], [True, True, False, True]]
                    elif mask[m,0] == True and mask[m,2] == True:
                        three_modals = [[True, True, True, False], [True, False, True, True]]
                    elif mask[m,0] == True and mask[m,3] == True:
                        three_modals = [[True, False, True, True], [True, True, False, True]]
                    elif mask[m,1] == True and mask[m,2] == True:
                        three_modals = [[True, True, True, False], [False, True, True, True]]
                    elif mask[m,1] == True and mask[m,3] == True:
                        three_modals = [[True, True, False, True], [False, True, True, True]]
                    elif mask[m,2] == True and mask[m,3] == True:
                        three_modals = [[True, False, True, True], [False, True, True, True]]
                    three_modals = torch.from_numpy(np.array(three_modals))
                    choice = np.random.choice(2, 1)
                    mask_teacher[m:m+1,:] = three_modals[choice]
                else:
                    mask_teacher[m:m+1,:] = torch.from_numpy(np.array([[True,True,True,True]]))

            mask_teacher = mask_teacher.cuda(non_blocking=True)

            model.module.is_training = True
            model_teacher.module.is_training = True

            fuse_pred, sep_preds, prm_preds, de_xs = model(x, mask, finetune = True)
            _, _, prm_pred_teacher, de_xs_teacher = model_teacher(x, mask, finetune = True)

            ###Loss compute
            di_loss = torch.zeros(1).cuda().float()
            mse_loss = torch.nn.MSELoss()
            for di in range(len(de_xs_teacher)):
                di_loss += mse_loss(de_xs[di],de_xs_teacher[di])

            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss               

            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss

            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()

            for pi in range(len(prm_pred_teacher)):
                prm_cross_loss += criterions.softmax_weighted_loss(prm_preds[pi], prm_pred_teacher[pi], num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_preds[pi], prm_pred_teacher[pi], num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss

            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss + prm_loss + di_loss
            else:
                loss = fuse_loss + sep_loss + prm_loss + di_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)
            writer.add_scalar('di_loss', di_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fcross:{:.4f}, fdice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'di_loss:{:.4f},'.format(di_loss.item())
            msg += 'scross:{:.4f}, sdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'pcross:{:.4f}, pdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))
        
        if (epoch+1) % 50 == 0 or (epoch>=(args.num_epochs-10)):
            file_name = os.path.join(ckpts, f'{args.mode}_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)

        if (epoch+1) % 50 == 0:
            model_teacher.load_state_dict(model.state_dict())
            print("The weights of the teacher model have been updated...")

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
