import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .data_utils import pkload

import numpy as np
import nibabel as nib

HGG = []
LGG = []
for i in range(0, 260):
    HGG.append(str(i).zfill(3))# zfill() 方法返回指定长度的字符串，原字符串右对齐，前面填充0。
for i in range(336, 370):
    HGG.append(str(i).zfill(3))
for i in range(260, 336):
    LGG.append(str(i).zfill(3))

mask_array = np.array([[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True], 
                      [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
                      [True, True, True, True]])

class Brats_loadall_nii_mixup(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt',scale=8):
        data_file_path = os.path.join(root, train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.len_of_org = len(self.volpaths)
        self.transforms = eval(transforms or 'Identity()')
        '''
         self.transforms = Compose([
         RandCrop3D((80,80,80)),                #将[1,4,240,240,155]的数据随机切割为[1,4,80,80,80]
         RandomRotion(10),                      #随机旋转0~10度
         RandomIntensityChange((0.1,0.1)),      #随机强度改变 x = x * [0.9~1.1] + [-0.1,0.1]
         RandomFlip(0),                         #随机翻转x,y,z轴
         NumpyType((np.float32, np.int64)),])   #将数据转为np.float32或np.int64，默认为np.float32类型
        '''
        self.names = datalist
        self.num_cls = num_cls
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])#默认执行此项
        
        assert scale>=1, "请确保scale不小于1"
        self.volpaths = self.volpaths*int(scale)
        self.names = datalist*int(scale)

        print("length of ORG training set is {}".format(self.len_of_org))
        print("length of AUG training set is {}".format(len(self.volpaths)))

    def aug_fun(self, x1, y1, x2, y2):
        H1,W1,Z1 = y1.shape                      
        H2,W2,Z2 = y2.shape

        H = max(H1,H2)              #取较大者，防止边界溢出
        W = max(W1,W2)
        Z = max(Z1,Z2)

        s1 = (H-H1)//2              
        s2 = (W-W1)//2
        s3 = (Z-Z1)//2

        t1 = (H-H2)//2
        t2 = (W-W2)//2
        t3 = (Z-Z2)//2

        new_x = np.zeros([H,W,Z, x1.shape[3]]).astype(np.float32)
        new_y1 = np.zeros([H,W,Z]).astype(np.int64)
        new_y1[s1:s1+H1,s2:s2+W1,s3:s3+Z1] = y1     #装载原始图像GT
        
        new_x.fill(x1.min())             #用原始图像的最小值(空白区域)填充
        new_x[s1:s1+H1,s2:s2+W1,s3:s3+Z1,:] = x1*0.5
        new_x[t1:t1+H2,t2:t2+W2,t3:t3+Z2,0] += x2[:,:,:,0]*0.5
        new_x[t1:t1+H2,t2:t2+W2,t3:t3+Z2,1] += x2[:,:,:,1]*0.5
        new_x[t1:t1+H2,t2:t2+W2,t3:t3+Z2,2] += x2[:,:,:,2]*0.5
        new_x[t1:t1+H2,t2:t2+W2,t3:t3+Z2,3] += x2[:,:,:,3]*0.5

        
        for i in range(0,H2):
            for j in range(0,W2):
                for k in range(0,Z2):#当某点在两张图像上均为肿瘤区域时，取等级较高的值
                    m = 0                                           #BG
                    if 3 in [new_y1[t1+i,t2+j,t3+k],y2[i,j,k]]:     #ET
                        m = 3
                    elif 1 in [new_y1[t1+i,t2+j,t3+k],y2[i,j,k]]:   #NCR/NET
                        m = 1
                    elif 2 in [new_y1[t1+i,t2+j,t3+k],y2[i,j,k]]:   #ED
                        m = 2
                    new_y1[t1+i,t2+j,t3+k] = m
        new_x = new_x[s1:s1+H1,s2:s2+W1,s3:s3+Z1,:]                 #移除padding
        new_y1 = new_y1[s1:s1+H1,s2:s2+W1,s3:s3+Z1]                 #移除padding

        return new_x, new_y1

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)

        if index >=self.len_of_org:
            # 数据增广
            while True:
                aug_index = np.random.choice(self.len_of_org)  # 随机配对
                if aug_index != index % self.len_of_org:
                    break
            aug_vol_path = self.volpaths[aug_index]
            # aug_diff_path = self.volpaths[aug_index].replace("vol", "diff")
            aug_seg_path = self.volpaths[aug_index].replace("vol", "seg")
            aug_x = np.load(aug_vol_path).astype(np.float32)
            # aug_diff = np.load(aug_diff_path).astype(np.float32)  #[H,W,Z,model_num]
            aug_y = np.load(aug_seg_path).astype(np.int64)   #[H,W,Z]

            x, y = self.aug_fun(x, y, aug_x, aug_y)
            # x, y = self.aug_fun(x, y, aug_diff, aug_y)

        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)



class Brats_loadall_nii_aug(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt',scale=8):
        data_file_path = os.path.join(root, train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.len_of_org = len(self.volpaths)
        self.transforms = eval(transforms or 'Identity()')
        '''
         self.transforms = Compose([
         RandCrop3D((80,80,80)),                #将[1,4,240,240,155]的数据随机切割为[1,4,80,80,80]
         RandomRotion(10),                      #随机旋转0~10度
         RandomIntensityChange((0.1,0.1)),      #随机强度改变 x = x * [0.9~1.1] + [-0.1,0.1]
         RandomFlip(0),                         #随机翻转x,y,z轴
         NumpyType((np.float32, np.int64)),])   #将数据转为np.float32或np.int64，默认为np.float32类型
        '''
        self.names = datalist
        self.num_cls = num_cls
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])#默认执行此项
        
        assert scale>=1, "请确保scale不小于1"
        self.volpaths = self.volpaths*int(scale)
        self.names = datalist*int(scale)

        print("length of ORG training set is {}".format(self.len_of_org))
        print("length of AUG training set is {}".format(len(self.volpaths)))

    def aug_fun(self, x, y1, diff, y2):
        H1,W1,Z1 = y1.shape                      
        H2,W2,Z2 = y2.shape

        H = max(H1,H2)              #取较大者，防止边界溢出
        W = max(W1,W2)
        Z = max(Z1,Z2)

        s1 = (H-H1)//2              
        s2 = (W-W1)//2
        s3 = (Z-Z1)//2

        t1 = (H-H2)//2
        t2 = (W-W2)//2
        t3 = (Z-Z2)//2

        new_x = np.zeros([H,W,Z, x.shape[3]]).astype(np.float32)
        new_y1 = np.zeros([H,W,Z]).astype(np.int64)
        new_y2 = np.int64(y2>0)                     #将添加差分图像的GT中大于0的值取1，方便做乘法
        new_y1[s1:s1+H1,s2:s2+W1,s3:s3+Z1] = y1     #装载原始图像GT
        
        new_x.fill(x.min())             #用原始图像的最小值(空白区域)填充
        new_x[s1:s1+H1,s2:s2+W1,s3:s3+Z1,:] = x
        new_x[t1:t1+H2,t2:t2+W2,t3:t3+Z2,0] += diff[:,:,:,0]*new_y2.astype(np.float32)
        new_x[t1:t1+H2,t2:t2+W2,t3:t3+Z2,1] += diff[:,:,:,1]*new_y2.astype(np.float32)
        new_x[t1:t1+H2,t2:t2+W2,t3:t3+Z2,2] += diff[:,:,:,2]*new_y2.astype(np.float32)
        new_x[t1:t1+H2,t2:t2+W2,t3:t3+Z2,3] += diff[:,:,:,3]*new_y2.astype(np.float32)

        
        for i in range(0,H2):
            for j in range(0,W2):
                for k in range(0,Z2):#当某点在两张图像上均为肿瘤区域时，取等级较高的值
                    m = 0                                           #BG
                    if 3 in [new_y1[t1+i,t2+j,t3+k],y2[i,j,k]]:     #ET
                        m = 3
                    elif 1 in [new_y1[t1+i,t2+j,t3+k],y2[i,j,k]]:   #NCR/NET
                        m = 1
                    elif 2 in [new_y1[t1+i,t2+j,t3+k],y2[i,j,k]]:   #ED
                        m = 2
                    new_y1[t1+i,t2+j,t3+k] = m
        new_x = new_x[s1:s1+H1,s2:s2+W1,s3:s3+Z1,:]                 #移除padding
        new_y1 = new_y1[s1:s1+H1,s2:s2+W1,s3:s3+Z1]                 #移除padding

        return new_x, new_y1

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)

        if index >=self.len_of_org:
            # 数据增广
            while True:
                aug_index = np.random.choice(self.len_of_org)  # 随机配对
                if aug_index != index % self.len_of_org:
                    break

            aug_diff_path = self.volpaths[aug_index].replace("vol", "diff")
            aug_seg_path = self.volpaths[aug_index].replace("vol", "seg")

            aug_diff = np.load(aug_diff_path).astype(np.float32)  #[H,W,Z,model_num]
            aug_y = np.load(aug_seg_path).astype(np.int64)   #[H,W,Z]

            x, y = self.aug_fun(x, y, aug_diff, aug_y)

        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)
        
class Brats_loadall_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt'):
        data_file_path = os.path.join(root, train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        '''
         self.transforms = Compose([
         RandCrop3D((80,80,80)),                #将[1,4,240,240,155]的数据随机切割为[1,4,80,80,80]
         RandomRotion(10),                      #随机旋转0~10度
         RandomIntensityChange((0.1,0.1)),      #随机强度改变 x = x * [0.9~1.1] + [-0.1,0.1]
         RandomFlip(0),                         #随机翻转x,y,z轴
         NumpyType((np.float32, np.int64)),])   #将数据转为np.float32或np.int64，默认为np.float32类型
        '''
        self.names = datalist
        self.num_cls = num_cls
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])#默认执行此项

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', test_file='test.txt'):
        data_file_path = os.path.join(root, test_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_val_nii(Dataset):
    def __init__(self, transforms='', root=None, settype='train', modal='all'):
        data_file_path = os.path.join(root, 'val.txt')
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        mask = mask_array[index%15]
        mask = torch.squeeze(torch.from_numpy(mask), dim=0)
        return x, y, mask, name

    def __len__(self):
        return len(self.volpaths)
