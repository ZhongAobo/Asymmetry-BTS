import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from layers import normalization
from layers import general_conv3d
from layers import prm_generator_laststage, prm_generator, region_aware_modal_fusion

basic_dims = 16
class Encoder(nn.Module):#编码器类
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = general_conv3d(1, basic_dims, pad_type='reflect')#nn.Conv3d(1,16,3,1,1,'reflect',bias='true')
        self.e1_c2 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')#nn.Conv3d(16,16,3,1,1,'reflect',bias='true')
        self.e1_c3 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')#nn.Conv3d(16,16,3,1,1,'reflect',bias='true')

        self.e2_c1 = general_conv3d(basic_dims, basic_dims*2, stride=2, pad_type='reflect')#nn.Conv3d(16,32,3,2,1,'reflect',bias='true')
        self.e2_c2 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')#nn.Conv3d(32,32,3,1,1,'reflect',bias='true')
        self.e2_c3 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')#nn.Conv3d(32,32,3,1,1,'reflect',bias='true')

        self.e3_c1 = general_conv3d(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')#nn.Conv3d(32,64,3,2,1,'reflect',bias='true')
        self.e3_c2 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')#nn.Conv3d(64,64,3,1,1,'reflect',bias='true')
        self.e3_c3 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')#nn.Conv3d(64,64,3,1,1,'reflect',bias='true')

        self.e4_c1 = general_conv3d(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')#nn.Conv3d(64,128,3,2,1,'reflect',bias='true')
        self.e4_c2 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')#nn.Conv3d(128,128,3,1,1,'reflect',bias='true')
        self.e4_c3 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')#nn.Conv3d(128,128,3,1,1,'reflect',bias='true')

    def forward(self, x):                   #x.Size([1, 1, 80, 80, 80])   这里的x为某一个模态的图像集
        x1 = self.e1_c1(x)                  #x1.Size([1, 16, 80, 80, 80])
        x1 = x1 + self.e1_c3(self.e1_c2(x1))#x1.Size([1, 16, 80, 80, 80])

        x2 = self.e2_c1(x1)                 #x2.Size([1, 32, 40, 40, 40])
        x2 = x2 + self.e2_c3(self.e2_c2(x2))#x2.Size([1, 32, 40, 40, 40])

        x3 = self.e3_c1(x2)                 #x3.Size([1, 64, 20, 20, 20])
        x3 = x3 + self.e3_c3(self.e3_c2(x3))#x3.Size([1, 64, 20, 20, 20])

        x4 = self.e4_c1(x3)                 #x4.Size([1, 128, 10, 10, 10])
        x4 = x4 + self.e4_c3(self.e4_c2(x4))#x4.Size([1, 128, 10, 10, 10])

        return x1, x2, x3, x4

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)#是每一行和为1

    def forward(self, x1, x2, x3, x4):
        de_x4 = self.d3_c1(self.d3(x4))#x4:[1,128,10,10,10]->[1,128,20,20,20]->[1,64,20,20,20]->de_x4

        cat_x3 = torch.cat((de_x4, x3), dim=1)#[1,128,20,20,20]
        de_x3 = self.d3_out(self.d3_c2(cat_x3))#[1,64,20,20,20]
        de_x3 = self.d2_c1(self.d2(de_x3))#[1,32,40,40,40]

        cat_x2 = torch.cat((de_x3, x2), dim=1)#[1,64,40,40,40]
        de_x2 = self.d2_out(self.d2_c2(cat_x2))#[1,32,40,40,40]
        de_x2 = self.d1_c1(self.d1(de_x2))#[1,16,80,80,80]

        cat_x1 = torch.cat((de_x2, x1), dim=1)#[1,32,80,80,80]
        de_x1 = self.d1_out(self.d1_c2(cat_x1))#[1,16,80,80,80]

        logits = self.seg_layer(de_x1)#[1,16,80,80,80]->[1,4,80,80,80]
        pred = self.softmax(logits)#[1,4,80,80,80]

        return pred #[1,4,80,80,80]

class Decoder_fuse(nn.Module):#解码器融合
    def __init__(self, num_cls=4):
        super(Decoder_fuse, self).__init__()

        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')#nn.Conv3d(128,64,3,1,1,'reflect',bias='true')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')#nn.Conv3d(128,64,3,1,1,'reflect',bias='true')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')#nn.Conv3d(64,64,1,1,0,'reflect',bias='true')

        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')#nn.Conv3d(64,32,3,1,1,'reflect',bias='true')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')#nn.Conv3d(64,32,3,1,1,'reflect',bias='true')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')#nn.Conv3d(32,32,1,1,0,'reflect',bias='true')

        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')#nn.Conv3d(32,16,3,1,1,'reflect',bias='true')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')#nn.Conv3d(32,16,3,1,1,'reflect',bias='true')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')#nn.Conv3d(16,16,1,1,0,'reflect',bias='true')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)#nn.Conv3d(16,4,1,1,0,'reflect',bias='true
        self.softmax = nn.Softmax(dim=1)#Softmax用于最后一层以对值进行归一化  np.exp(x) / np.sum(np.exp(x)
        #对于三维的输入张量，可选用最近邻 (nearest neighbor)
        #对于四维的输入张量，可选用线性(linear)、双线性(bilinear)、双三次(bicubic)
        #对于五维的输入张量，可选用三线性(trilinear)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

        self.RFM4 = region_aware_modal_fusion(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = region_aware_modal_fusion(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = region_aware_modal_fusion(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = region_aware_modal_fusion(in_channel=basic_dims*1, num_cls=num_cls)

        self.prm_generator4 = prm_generator_laststage(in_channel=basic_dims*8, num_cls=num_cls)
        self.prm_generator3 = prm_generator(in_channel=basic_dims*4, num_cls=num_cls)
        self.prm_generator2 = prm_generator(in_channel=basic_dims*2, num_cls=num_cls)
        self.prm_generator1 = prm_generator(in_channel=basic_dims*1, num_cls=num_cls)
        #概率图就是每个像素点在区域维度上相加为1，其中值最大的区域就是该点被分类的区域

    def forward(self, x1, x2, x3, x4, mask):
        prm_pred4 = self.prm_generator4(x4, mask)#x4:[1,4,128,10,10,10] prm_pred4:[1,num_cls=4,10,10,10]
        # print("++++++++++{},{},{}++++++++++".format(x4.shape,mask,prm_pred4.shape))
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask)#detach()用于切断相应参数的反向传播 [1,128,10,10,10]
        de_x4 = self.d3_c1(self.up2(de_x4))#de_x4:[1,64,20,20,20]

        prm_pred3 = self.prm_generator3(de_x4, x3, mask)#x3[1,4,64,20,20,20] prm_pred3:[1,num_cls=4,20,20,20]
        de_x3 = self.RFM3(x3, prm_pred3.detach(), mask)#[1, 64, 20, 20, 20]
        de_x3 = torch.cat((de_x3, de_x4), dim=1)#[1,128,20,20,20]
        de_x3 = self.d3_out(self.d3_c2(de_x3))#de_x3:[1,128,20,20,20]->[1,64,20,20,20]->[1,64,20,20,20]
        de_x3 = self.d2_c1(self.up2(de_x3))#de_x3:[1,64,20,20,20]->[1,64,40,40,40]->[1,32,40,40,40]

        prm_pred2 = self.prm_generator2(de_x3, x2, mask)#x2[1,4,32,40,40,40] prm_pred2:[1,num_cls=4,40,40,40]
        de_x2 = self.RFM2(x2, prm_pred2.detach(), mask)#[1, 32, 40, 40, 40]
        de_x2 = torch.cat((de_x2, de_x3), dim=1)#de_x2:[1, 32, 40, 40, 40]->[1,64,40,40,40]
        de_x2 = self.d2_out(self.d2_c2(de_x2))#de_x2:[1,64,40,40,40]->[1,32,40,40,40]->[1,32,40,40,40]
        de_x2 = self.d1_c1(self.up2(de_x2))#de_x2:[1,32,40,40,40]->[1,32,80,80,80]->[1,16,80,80,80]

        prm_pred1 = self.prm_generator1(de_x2, x1, mask)#x1[1,4,16,80,80,80] prm_pred1[1,num_cls=4,80,80,80]
        de_x1 = self.RFM1(x1, prm_pred1.detach(), mask)#de_x1:[1,16,80,80,80]
        de_x1 = torch.cat((de_x1, de_x2), dim=1)#de_x1:[1,16,80,80,80]->[1,32,80,80,80]
        de_x1 = self.d1_out(self.d1_c2(de_x1))#de_x1:[1,32,80,80,80]->[1,16,80,80,80]->[1,16,80,80,80]

        logits = self.seg_layer(de_x1)#[1,16,80,80,80]->[1,4,80,80,80]
        pred = self.softmax(logits)#[1,4,80,80,80]

        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4))#[1,4,80,80,80]([1,4,80,80,80],[1,4,80,80,80],[1,4,80,80,80],[1,4,80,80,80])

class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()
        # self.shareEncoder = Encoder()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.is_training = False

        for m in self.modules():#初始化权重
            if isinstance(m, nn.Conv3d):#isinstance(object, classinfo) 如果参数object是classinfo的实例或子类的实例，返回True，否则False。
                torch.nn.init.kaiming_normal_(m.weight) #初始默认权重服从0为中心的正态分布

    def forward(self, x, mask):#mask就是四个模态的使用情况，例如[True,True,False,False]
        #extract feature from different layers   从不同层中提取特征   x.Size([1, 4, 80, 80, 80])
        # flair_x1, flair_x2, flair_x3, flair_x4 = self.shareEncoder(x[:, 0:1, :, :, :])#取出第一个模态
        # t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self.shareEncoder(x[:, 1:2, :, :, :])#取出第二个模态
        # t1_x1, t1_x2, t1_x3, t1_x4 = self.shareEncoder(x[:, 2:3, :, :, :])#取出第三个模态
        # t2_x1, t2_x2, t2_x3, t2_x4 = self.shareEncoder(x[:, 3:4, :, :, :])#取出第四个模态
        flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, :, :, :])#取出第一个模态
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self.t1ce_encoder(x[:, 1:2, :, :, :])#取出第二个模态
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, :, :, :])#取出第三个模态
        t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, :, :, :])#取出第四个模态
        #[1, 16, 80, 80, 80][1, 32, 40, 40, 40][1, 64, 20, 20, 20][1, 128, 10, 10, 10]

        x1 = torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1) #Bx4xCxHWZ
        #in:[1, 16, 80, 80, 80] out:[1,4,16,80,80,80]
        x2 = torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1)
        #in:[1, 32, 40, 40, 40] out:[1,4,32,40,40,40]
        x3 = torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1)
        #in:[1, 64, 20, 20, 20] out:[1,4,64,20,20,20]
        x4 = torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1)
        #in:[1, 128, 10, 10, 10] out:[1,4,128,10,10,10]
        
        fuse_pred, prm_preds = self.decoder_fuse(x1, x2, x3, x4, mask)#[1,4,80,80,80] ([1,4,80,80,80],[1,4,80,80,80],[1,4,80,80,80],[1,4,80,80,80])

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4)#flair_pred:[1,4,80,80,80]
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4)#t1ce_pred:[1,4,80,80,80]
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4)#t1_pred:[1,4,80,80,80]
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4)#t2_pred:[1,4,80,80,80]
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds #[1,4,80,80,80] 4*[1,4,80,80,80] 4*[1,4,80,80,80]
        return fuse_pred#[1,4,80,80,80]
