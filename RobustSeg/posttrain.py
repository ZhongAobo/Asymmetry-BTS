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

import models
from data.transforms import *
from data.datasets_aug import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from predict import AverageMeter, test_softmax


import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default='/home/SHARE2/ZXY/RFNet/BRATS2020_Training_none_npy', type=str)
parser.add_argument('--dataname', default='BRATS2020', type=str)
parser.add_argument('--savepath', default='/home/SHARE2/ZXY/RFNet/RF2Robust/results_distill', type=str)
parser.add_argument('--resume', default='/home/SHARE2/ZXY/RFNet/RF2Robust/results_distill/model_300.pth', type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--seed', default=7788, type=int)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###modality missing mask 
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
# masks = [[True,True,True,True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1ce', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
# mask_name = ['flairt1cet1t2']

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
    model_teacher = models.Model()
    model = torch.nn.DataParallel(model).cuda()
    model_teacher = torch.nn.DataParallel(model_teacher).cuda()


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
        model_teacher.load_state_dict(checkpoint['state_dict'])

    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = args.iter_per_epoch
    train_iter = iter(train_loader)

    for epoch in range(args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
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
            model_teacher.module.is_training = True
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

            outputs_teacher = model(x, mask_teacher)
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

            DisLoss = torch.zeros(1).cuda().float()

            DisLoss += mse_loss(outputs['content_flair']['s1'],outputs_teacher['content_flair']['s1'])
            DisLoss += mse_loss(outputs['content_flair']['s2'],outputs_teacher['content_flair']['s2'])
            DisLoss += mse_loss(outputs['content_flair']['s3'],outputs_teacher['content_flair']['s3'])
            DisLoss += mse_loss(outputs['content_flair']['s4'],outputs_teacher['content_flair']['s4'])
            
            DisLoss += mse_loss(outputs['content_t1c']['s1'],outputs_teacher['content_t1c']['s1'])
            DisLoss += mse_loss(outputs['content_t1c']['s2'],outputs_teacher['content_t1c']['s2'])
            DisLoss += mse_loss(outputs['content_t1c']['s3'],outputs_teacher['content_t1c']['s3'])
            DisLoss += mse_loss(outputs['content_t1c']['s4'],outputs_teacher['content_t1c']['s4'])

            DisLoss += mse_loss(outputs['content_t1']['s1'],outputs_teacher['content_t1']['s1'])
            DisLoss += mse_loss(outputs['content_t1']['s2'],outputs_teacher['content_t1']['s2'])
            DisLoss += mse_loss(outputs['content_t1']['s3'],outputs_teacher['content_t1']['s3'])
            DisLoss += mse_loss(outputs['content_t1']['s4'],outputs_teacher['content_t1']['s4'])

            DisLoss += mse_loss(outputs['content_t2']['s1'],outputs_teacher['content_t2']['s1'])
            DisLoss += mse_loss(outputs['content_t2']['s2'],outputs_teacher['content_t2']['s2'])
            DisLoss += mse_loss(outputs['content_t2']['s3'],outputs_teacher['content_t2']['s3'])
            DisLoss += mse_loss(outputs['content_t2']['s4'],outputs_teacher['content_t2']['s4'])



            loss  = ce_loss + dice_loss + l2_loss + 0.1*rec_loss + 0.1*kl_loss + DisLoss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'ce_loss:{:.4f},'.format(ce_loss.item())
            msg += 'dice_loss:{:.4f},'.format(dice_loss.item())
            msg += 'l2_loss:{:.4f},'.format(l2_loss.item())
            msg += 'rec_loss:{:.4f}, kl_loss:{:.4f},'.format(0.1*rec_loss.item(), 0.1*kl_loss.item())
            msg += ' DisLoss:{:.4f},'.format(DisLoss.item())
            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))

        ##########model save
        if (epoch+1) % 50 == 0:
            model_teacher.load_state_dict(model.state_dict())
            print("The weights of the teacher model have been updated...")
            
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            },
            file_name)
        
        if (epoch+1) % 50 == 0 or (epoch>=args.num_epochs-3):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+301))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)
        
    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    #########Evaluate the last epoch model
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
