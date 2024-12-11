import torch
import torch.nn as nn

def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
        """
        batch方向做归一化，算HWZ的均值，对小batchsize效果不好；BN主要缺点是对batchsize的大小比较敏感，
        由于每次计算均值和方差是在一个batch上，所以如果batchsize太小，则计算的均值、方差不足以代表整个数据分布
        """
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
        """
        将channel方向分group，然后每个group内做归一化，算(C//G)HW的均值；这样与batchsize无关，不受其约束。
        """
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
        """
        一个channel内做归一化，算H*W的均值，用在风格化迁移；因为在图像风格化中，生成结果主要依赖于某个图像实例，
        所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。
        """
    elif norm == 'sync_bn':
        m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class general_conv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='reflect', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class prm_generator_laststage(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(prm_generator_laststage, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel*4, int(in_channel//4), k_size=1, padding=0, stride=1),#nn.Conv3d(4*in,in/4,1,1,0)
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),#nn.Conv3d(in/4,in/4,3,1,1)
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))#nn.Conv3d(in/4,in,1,1,0)

        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel, 16, k_size=1, stride=1, padding=0),#nn.Conv3d(in,16,1,1,0)
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),#nn.Conv3d(16,4,1,1,0)
                            nn.Softmax(dim=1))#每个像素点在区域维度上相加为1，其中值最大的区域就是该点被分类的区域

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)#生成与括号内的变量维度维度一致的全是零的内容。
        y[mask, ...] = x[mask, ...]#y只在mask为True的模态与x相应的模态相等，其余的模态均为0 mask:[1,4]
        y = y.view(B, -1, H, W, Z)

        seg = self.prm_layer(self.embedding_layer(y))
        return seg

class prm_generator(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(prm_generator, self).__init__()

        self.embedding_layer = nn.Sequential(
                            general_conv3d(in_channel*4, int(in_channel//4), k_size=1, padding=0, stride=1),#nn.Conv3d(4*in,in/4,1,1,0)
                            general_conv3d(int(in_channel//4), int(in_channel//4), k_size=3, padding=1, stride=1),#nn.Conv3d(in/4,in/4,3,1,1)
                            general_conv3d(int(in_channel//4), in_channel, k_size=1, padding=0, stride=1))#nn.Conv3d(in/4,in,1,1,0)

        self.prm_layer = nn.Sequential(
                            general_conv3d(in_channel*2, 16, k_size=1, stride=1, padding=0),#nn.Conv3d(2*in,16,1,1,0)
                            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),#nn.Conv3d(16,4,1,1,0)
                            nn.Softmax(dim=1))#每个像素点在区域维度上相加为1，其中值最大的区域就是该点被分类的区域

    def forward(self, x1, x2, mask):
        B, K, C, H, W, Z = x2.size()
        y = torch.zeros_like(x2)
        y[mask, ...] = x2[mask, ...]
        y = y.view(B, -1, H, W, Z)

        seg = self.prm_layer(torch.cat((x1, self.embedding_layer(y)), dim=1))
        return seg

####modal fusion in each region
class modal_fusion(nn.Module):#attention module in Fig 5
    def __init__(self, in_channel=64):
        super(modal_fusion, self).__init__()
        self.weight_layer = nn.Sequential(
                            nn.Conv3d(4*in_channel+1, 128, 1, padding=0, bias=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv3d(128, 4, 1, padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prm, region_name):#x:*[1, modal_num, feature_num,H,W,Z] prm::[1, 1, feature_num, H, W, Z]
        B, K, C, H, W, Z = x.size()#[1,modal=4,C,H,W,Z]

        prm_avg = torch.mean(prm, dim=(3,4,5), keepdim=False) + 1e-7    #[1,1,C] 实际上prm_avg中包含了C个相同的均值（表示该种区域在所有像素点中的占比）
        #feat_avg = torch.mean(x, dim=(3,4,5), keepdim=False) / prm_avg  #[1,modal=4,C] 获得用于生成注意权重的特征
        feat_avg = torch.mean(x, dim=(3,4,5), keepdim=False) 

        feat_avg = feat_avg / prm_avg  #[1,modal=4,C] 获得用于生成注意权重的特征
        #|-------4种模态通过不同特征对该种区域的贡献------|---归一化---|
        feat_avg = feat_avg.view(B, K*C, 1, 1, 1)#feat_avg[1,K*C,1,1,1] 归一化后的平均特征进行变形
        feat_avg = torch.cat((feat_avg, prm_avg[:, 0, 0, ...].view(B, 1, 1, 1, 1)), dim=1)#feat_avg[1,4*C+1,1,1,1] %%%%%%%%难以理解%%%%%%%%
        weight = torch.reshape(self.weight_layer(feat_avg), (B, K, 1))#[1,K*C+1,1,1,1]->[1,128,1,1,1]->[1,4,1,1,1]->weight[1,4,1]
        weight = self.sigmoid(weight).view(B, K, 1, 1, 1, 1)#weght[1,4,1,1,1,1]，weight就是不同模态对某一区域的权重

        ###we find directly using weighted sum still achieve competing performance 我们发现直接使用加权和仍然可以达到竞争性能
        region_feat = torch.sum(x * weight, dim=1)
        return region_feat#[1,C,H,W,Z] 返回的是某一区域的特征划分图

###fuse region feature
class region_fusion(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(region_fusion, self).__init__()
        self.fusion_layer = nn.Sequential(
                        general_conv3d(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d(in_channel, in_channel//2, k_size=1, padding=0, stride=1))

    def forward(self, x):
        B, _, _, H, W, Z = x.size()#[1,num_cls,C,H,W,Z]
        x = torch.reshape(x, (B, -1, H, W, Z))#[1,num_cls*C,H,W,Z]
        return self.fusion_layer(x)#[1,num_cls*C,H,W,Z]->[1,C,H,W,Z]->[1,C,H,W,Z]->[1,C/2,H,W,Z]

class region_aware_modal_fusion(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):#in_channel = 128/64/32/16
        super(region_aware_modal_fusion, self).__init__()
        self.num_cls = num_cls

        self.modal_fusion = nn.ModuleList([modal_fusion(in_channel=in_channel) for i in range(num_cls)])
        self.region_fusion = region_fusion(in_channel=in_channel, num_cls=num_cls)
        self.short_cut = nn.Sequential(
                        # general_conv3d(in_channel*num_cls, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel*4, in_channel, k_size=1, padding=0, stride=1),
                        general_conv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
                        general_conv3d(in_channel, in_channel//2, k_size=1, padding=0, stride=1))

        self.clsname_list = ['BG', 'NCR/NET', 'ED', 'ET'] ##BRATS2020 and BRATS2018
        self.clsname_list = ['BG', 'NCR', 'ED', 'NET', 'ET'] ##BRATS2015

    def forward(self, x, prm, mask):#prm:[1,4,H,W,Z],概率图,指示每个位置的脑肿瘤结构（包括健康的脑区域）的概率
        B, K, C, H, W, Z = x.size() 
        y = torch.zeros_like(x)#生成与括号内的变量维度维度一致的全是零的内容。y:[1,K,C,H,W,Z]
        y[mask, ...] = x[mask, ...]#mask:[1,4],包含至多4个模态，缺失的模态置为0

        prm = torch.unsqueeze(prm, 2).repeat(1, 1, C, 1, 1, 1)#prm:[1,num_cls,C,H,W,Z]
        ###divide modal features into different regions 在概率图的帮助下，RFM 成功地将多模态特征划分为不同的区域
        flair = y[:, 0:1, ...] * prm#[1,num_cls,C,H,W,Z]   特征划分是通过将特征与概率图相乘来实现的
        t1ce = y[:, 1:2, ...] * prm#[1,num_cls,C,H,W,Z]
        t1 = y[:, 2:3, ...] * prm#[1,num_cls,C,H,W,Z]
        t2 = y[:, 3:4, ...] * prm#[1,num_cls,C,H,W,Z]

        modal_feat = torch.stack((flair, t1ce, t1, t2), dim=1)#modal_feat:[1,4,num_cls=5,C,H,W,Z] 4个模态的各自区域的划分情况
        region_feat = [modal_feat[:, :, i, :, :] for i in range(self.num_cls)]#num_cls*[1,4,C,H,W,Z]   num_cls种区域通过四种模态的划分情况
        ###modal fusion in each region  各区域模态融合
        region_fused_feat = []
        for i in range(self.num_cls):#([1,C,H,W,Z],[1,C,H,W,Z],[1,C,H,W,Z],[1,C,H,W,Z])
            region_fused_feat.append(self.modal_fusion[i](region_feat[i], prm[:, i:i+1, ...], self.clsname_list[i]))#将不同模态间同一区域的特征进行融合
        region_fused_feat = torch.stack(region_fused_feat, dim=1)#[1,4,C,H,W,Z]
        '''
        region_fused_feat = torch.stack((self.modal_fusion[0](region_feat[0], prm[:, 0:1, ...], 'BG'),        将BG的概率图和BG的划分情况进行融合
                                         self.modal_fusion[1](region_feat[1], prm[:, 1:2, ...], 'NCR/NET'),   将NCR/NET的概率图和NCR/NET的划分情况进行融合
                                         self.modal_fusion[2](region_feat[2], prm[:, 2:3, ...], 'ED'),        将ED的概率图和ED的划分情况进行融合
                                         self.modal_fusion[3](region_feat[3], prm[:, 3:4, ...], 'ET')), dim=1)将ET的概率图和ET的划分情况进行融合
        region_fused_feat表示的是每种区域在当前分辨率下，用不同模态观测得到的的融合特征
        '''

        ###gain final feat with a short cut 通过short cut获得最终特征
        final_feat = torch.cat((self.region_fusion(region_fused_feat), self.short_cut(y.view(B, -1, H, W, Z))), dim=1)
        '''
        y表示所有可用模态 
        y[1,K,C,H,W,Z]->[1,K*C,H,W,Z]->[1,C,H,W,Z]->[1,C,H,W,Z]->[1,C/2,H,W,Z]
        region_fusion[1,K,C,H,W,Z]->[1,K*C,H,W,Z]->[1,C,H,W,Z]->[1,C,H,W,Z]->[1,C/2,H,W,Z]
        torch.cat(([1,C/2,H,W,Z],[1,C/2,H,W,Z]),dim=1)
        '''
        return final_feat#[1,C,H,W,Z]
