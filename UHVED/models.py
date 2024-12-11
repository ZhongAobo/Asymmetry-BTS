from os import setegid
from re import I
import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import general_conv3d,normalization
MODALITIES = ['Flair', 'T1c', 'T1', 'T2', 'seg']
HIDDEN_SPACE = 512
NB_CONV = 8
num_cls = 4
class ResBlock(nn.Module):
    def __init__(self,
                in_ch,
                 out_ch,
                 kernels=3,
                 acti_func='leakyrelu',
                 encoding=False,
                 double_n = True,
                 with_res=True,
                 stride=1):
        super(ResBlock, self).__init__()
        self.out_ch = out_ch
        self.acti_func = acti_func
        self.with_res = with_res

        self.stride = stride
        self.encoding = encoding
        self.double_n = double_n

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if self.encoding:
            self.conv_0 = nn.Conv3d(in_ch,out_ch,3,1,1)
            self.conv_1 = nn.Conv3d(out_ch,out_ch,3,1,1)
            self.in_0 = normalization(in_ch,'in')
            self.in_1 = normalization(out_ch,'in')
        else:
            if self.double_n:
                self.conv_0 = nn.Conv3d(in_ch,out_ch,3,1,1)
                self.conv_1 = nn.Conv3d(out_ch,out_ch//2,3,1,1)   
                self.in_0 = normalization(in_ch,'in')    
                self.in_1 = normalization(out_ch,'in')
            else:
                self.conv_0 = nn.Conv3d(in_ch,out_ch//2,3,1,1)
                self.conv_1 = nn.Conv3d(out_ch//2,out_ch//2,3,1,1)         
                self.in_0 = normalization(in_ch,'in')
                self.in_1 = normalization(out_ch//2,'in')

    def forward(self,input_tensor):  
        output_tensor = self.in_0(input_tensor)
        output_tensor = self.lrelu(output_tensor)
        output_tensor = self.conv_0(output_tensor)

        output_tensor = self.in_1(output_tensor)
        output_tensor = self.lrelu(output_tensor)
        output_tensor = self.conv_1(output_tensor)
        return output_tensor

class ConvDecoderImg(nn.Module):
    def __init__(self):
        super(ConvDecoderImg, self).__init__()
        self.up_x2 = nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True)

        self.last_conv_Flair = general_conv3d(4,1,1,1,0,act_type=None)
        self.last_conv_T1c = general_conv3d(4,1,1,1,0,act_type=None)
        self.last_conv_T1 = general_conv3d(4,1,1,1,0,act_type=None)
        self.last_conv_T2 = general_conv3d(4,1,1,1,0,act_type=None)
        self.last_conv_seg = general_conv3d(4,num_cls,1,1,0,act_type=None)

        self.res_Flair_0 = ResBlock(48,32,3,'leakyrelu',False,True)
        self.res_Flair_1 = ResBlock(24,16,3,'leakyrelu',False,True)
        self.res_Flair_2 = ResBlock(12,8,3,'leakyrelu',False,True)

        self.res_T1c_0 = ResBlock(48,32,3,'leakyrelu',False,True)
        self.res_T1c_1 = ResBlock(24,16,3,'leakyrelu',False,True)
        self.res_T1c_2 = ResBlock(12,8,3,'leakyrelu',False,True)

        self.res_T1_0 = ResBlock(48,32,3,'leakyrelu',False,True)
        self.res_T1_1 = ResBlock(24,16,3,'leakyrelu',False,True)
        self.res_T1_2 = ResBlock(12,8,3,'leakyrelu',False,True)

        self.res_T2_0 = ResBlock(48,32,3,'leakyrelu',False,True)
        self.res_T2_1 = ResBlock(24,16,3,'leakyrelu',False,True)
        self.res_T2_2 = ResBlock(12,8,3,'leakyrelu',False,True)

        self.res_seg_0 = ResBlock(48,32,3,'leakyrelu',False,True)
        self.res_seg_1 = ResBlock(24,16,3,'leakyrelu',False,True)
        self.res_seg_2 = ResBlock(12,8,3,'leakyrelu',False,True)

        self.softmax = nn.Softmax(dim=1)


    def forward(self,list_skips,is_inference):#{[B,4,112,112,112],[B,8,56,56,56],[B,16,28,28,28],[B,32,14,14,14]}
        flow = dict()
        list_skips = list_skips[::-1]
        
        if is_inference:
            flow_mod = list_skips[0]
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[1]),dim=1)
            flow_mod = self.res_seg_0(flow_mod)

            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[2]),dim=1)
            flow_mod = self.res_seg_1(flow_mod)

            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[3]),dim=1)
            flow_mod = self.res_seg_2(flow_mod)

            flow['seg'] = self.last_conv_seg(flow_mod)
        else:
            flow_mod = list_skips[0]
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[1]),dim=1)
            flow_mod = self.res_Flair_0(flow_mod)
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[2]),dim=1)
            flow_mod = self.res_Flair_1(flow_mod)
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[3]),dim=1)
            flow_mod = self.res_Flair_2(flow_mod)
            flow['Flair'] = self.last_conv_Flair(flow_mod)

            flow_mod = list_skips[0]
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[1]),dim=1)
            flow_mod = self.res_T1c_0(flow_mod)
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[2]),dim=1)
            flow_mod = self.res_T1c_1(flow_mod)
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[3]),dim=1)
            flow_mod = self.res_T1c_2(flow_mod)
            flow['T1c'] = self.last_conv_T1c(flow_mod)

            flow_mod = list_skips[0]
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[1]),dim=1)
            flow_mod = self.res_T1_0(flow_mod)
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[2]),dim=1)
            flow_mod = self.res_T1_1(flow_mod)
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[3]),dim=1)
            flow_mod = self.res_T1_2(flow_mod)
            flow['T1'] = self.last_conv_T1(flow_mod)

            flow_mod = list_skips[0]
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[1]),dim=1)
            flow_mod = self.res_T2_0(flow_mod)
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[2]),dim=1)
            flow_mod = self.res_T2_1(flow_mod)
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[3]),dim=1)
            flow_mod = self.res_T2_2(flow_mod)
            flow['T2'] = self.last_conv_T2(flow_mod)

            flow_mod = list_skips[0]
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[1]),dim=1)
            flow_mod = self.res_seg_0(flow_mod)
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[2]),dim=1)
            flow_mod = self.res_seg_1(flow_mod)
            flow_mod = self.up_x2(flow_mod)
            flow_mod = torch.cat((flow_mod,list_skips[3]),dim=1)
            flow_mod = self.res_seg_2(flow_mod)
            flow['seg'] = self.last_conv_seg(flow_mod)
            
        flow['seg'] = self.softmax(flow['seg'])
        return flow #[Flair:[B,1,112,112,112],T1c:[B,1,112,112,112],T1:[B,1,112,112,112],T2:[B,1,112,112,112],seg:[B,4,112,112,112]]
class GaussianSampler(nn.Module):
    def __init__(self):
        super(GaussianSampler, self).__init__()
    '''
            post_param{
    id=0        {mu:{Flair:[B,4,112,112,112],T1c:[B,4,112,112,112],T1:[B,4,112,112,112],T2:[B,4,112,112,112]},
                logvar:{Flair:[B,4,112,112,112],T1c:[B,4,112,112,112],T1:[B,4,112,112,112],T2:[B,4,112,112,112]}},

    id=1        {mu:{Flair:[B,8,56,56,56],T1c:[B,8,56,56,56],T1:[B,8,56,56,56],T2:[B,8,56,56,56]},
                logvar:{Flair:[B,8,56,56,56],T1c:[B,8,56,56,56],T1:[B,8,56,56,56],T2:[B,8,56,56,56]}},

    id=2        {mu:{Flair:[B,16,28,28,28],T1c:[B,16,28,28,28],T1:[B,16,28,28,28],T2:[B,16,28,28,28]},
                logvar:{Flair:[B,16,28,28,28],T1c:[B,16,28,28,28],T1:[B,16,28,28,28],T2:[B,16,28,28,28]}},

    id=3        {mu:{Flair:[B,32,14,14,14],T1c:[B,32,14,14,14],T1:[B,32,14,14,14],T2:[B,32,14,14,14]},
                logvar:{Flair:[B,32,14,14,14],T1c:[B,32,14,14,14],T1:[B,32,14,14,14],T2:[B,32,14,14,14]}}             
            }
    '''
    def forward(self, means, logvars, list_mod, choices, is_inference):
        mu_prior = torch.zeros(means[list_mod[0]].shape).cuda()        #[B,4,112,112,112]
        log_prior = torch.zeros(means[list_mod[0]].shape).cuda()       #[B,4,112,112,112]
        eps = 1e-7

        a = [1/torch.exp(logvars[mod]) + eps for mod in list_mod]
        b = [means[mod]/(torch.exp(logvars[mod]) + eps) for mod in list_mod]
        T = torch.zeros(4,mu_prior.shape[0],mu_prior.shape[1],mu_prior.shape[2],mu_prior.shape[3],mu_prior.shape[4]).cuda()
        mu = torch.zeros(4,log_prior.shape[0],log_prior.shape[1],log_prior.shape[2],log_prior.shape[3],log_prior.shape[4]).cuda()
        for i in range(4):
            if choices[0][i]:
                T[i,...] = a[i]
                mu[i,...] = b[i]

        T = torch.cat([T,(1+log_prior).unsqueeze(0)],0)  #{(2~5)*[B,4,112,112,112]}
        mu = torch.cat([mu,(mu_prior).unsqueeze(0)],0)   #{(2~5)*[B,4,112,112,112]}

        posterior_means = torch.sum(mu,0) / torch.sum(T,0)  #[B,4,112,112,112]
        var = 1 / torch.sum(T,0)                            #[B,4,112,112,112]
        posterior_logvars = torch.log(var+eps)             #[B,4,112,112,112]

        if is_inference:
            return posterior_means                          #[B,4,112,112,112]
        else:
            noise_sample = torch.randn(posterior_means.shape).cuda()   #[B,4,112,112,112]
            return posterior_means + torch.exp(0.5*posterior_logvars)*noise_sample  #[B,4,112,112,112]
class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.skip_ind = [1, 3, 5, 7]
        self.hidden = [NB_CONV//2,NB_CONV,NB_CONV*2,NB_CONV*4]
        self.list_skip_flow = [{'mu':dict(), 'logvar':dict()} for k in range(len(self.skip_ind))]

        self.cnn_Flair_0 = general_conv3d(1,NB_CONV,1,1,0,pad_type='reflect',bias=True)
        self.cnn_Flair_1 = ResBlock(NB_CONV,NB_CONV,3,'leakyrelu',True)
        self.cnn_Flair_2 = nn.MaxPool3d(2,2)
        self.cnn_Flair_3 = ResBlock(NB_CONV,NB_CONV*2,3,'leakyrelu',True)
        self.cnn_Flair_4 = nn.MaxPool3d(2,2)
        self.cnn_Flair_5 = ResBlock(NB_CONV*2,NB_CONV*4,3,'leakyrelu',True)
        self.cnn_Flair_6 = nn.MaxPool3d(2,2)
        self.cnn_Flair_7 = ResBlock(NB_CONV*4,NB_CONV*8,3,'leakyrelu',True)

        self.cnn_T1c_0 = general_conv3d(1,NB_CONV,1,1,0,pad_type='reflect',bias=True)
        self.cnn_T1c_1 = ResBlock(NB_CONV,NB_CONV,3,'leakyrelu',True)
        self.cnn_T1c_2 = nn.MaxPool3d(2,2)
        self.cnn_T1c_3 = ResBlock(NB_CONV,NB_CONV*2,3,'leakyrelu',True)
        self.cnn_T1c_4 = nn.MaxPool3d(2,2)
        self.cnn_T1c_5 = ResBlock(NB_CONV*2,NB_CONV*4,3,'leakyrelu',True)
        self.cnn_T1c_6 = nn.MaxPool3d(2,2)
        self.cnn_T1c_7 = ResBlock(NB_CONV*4,NB_CONV*8,3,'leakyrelu',True)

        self.cnn_T1_0 = general_conv3d(1,NB_CONV,1,1,0,pad_type='reflect',bias=True)
        self.cnn_T1_1 = ResBlock(NB_CONV,NB_CONV,3,'leakyrelu',True)
        self.cnn_T1_2 = nn.MaxPool3d(2,2)
        self.cnn_T1_3 = ResBlock(NB_CONV,NB_CONV*2,3,'leakyrelu',True)
        self.cnn_T1_4 = nn.MaxPool3d(2,2)
        self.cnn_T1_5 = ResBlock(NB_CONV*2,NB_CONV*4,3,'leakyrelu',True)
        self.cnn_T1_6 = nn.MaxPool3d(2,2)
        self.cnn_T1_7 = ResBlock(NB_CONV*4,NB_CONV*8,3,'leakyrelu',True)

        self.cnn_T2_0 = general_conv3d(1,NB_CONV,1,1,0,pad_type='reflect',bias=True)
        self.cnn_T2_1 = ResBlock(NB_CONV,NB_CONV,3,'leakyrelu',True)
        self.cnn_T2_2 = nn.MaxPool3d(2,2)
        self.cnn_T2_3 = ResBlock(NB_CONV,NB_CONV*2,3,'leakyrelu',True)
        self.cnn_T2_4 = nn.MaxPool3d(2,2)
        self.cnn_T2_5 = ResBlock(NB_CONV*2,NB_CONV*4,3,'leakyrelu',True)
        self.cnn_T2_6 = nn.MaxPool3d(2,2)
        self.cnn_T2_7 = ResBlock(NB_CONV*4,NB_CONV*8,3,'leakyrelu',True)        

    def forward(self,images):
        flow_mod = images['Flair']
        flow_mod = self.cnn_Flair_0(flow_mod)
        flow_mod = self.cnn_Flair_1(flow_mod)
        self.list_skip_flow[0]['mu']['Flair'] = flow_mod[:,:self.hidden[0],:,:,:]
        self.list_skip_flow[0]['logvar']['Flair'] = flow_mod[:,self.hidden[0]:,:,:,:]
        flow_mod = self.cnn_Flair_2(flow_mod)
        flow_mod = self.cnn_Flair_3(flow_mod)
        self.list_skip_flow[1]['mu']['Flair'] = flow_mod[:,:self.hidden[1],:,:,:]
        self.list_skip_flow[1]['logvar']['Flair'] = flow_mod[:,self.hidden[1]:,:,:,:]
        flow_mod = self.cnn_Flair_4(flow_mod)
        flow_mod = self.cnn_Flair_5(flow_mod)
        self.list_skip_flow[2]['mu']['Flair'] = flow_mod[:,:self.hidden[2],:,:,:]
        self.list_skip_flow[2]['logvar']['Flair'] = flow_mod[:,self.hidden[2]:,:,:,:]
        flow_mod = self.cnn_Flair_6(flow_mod)
        flow_mod = self.cnn_Flair_7(flow_mod)
        self.list_skip_flow[3]['mu']['Flair'] = flow_mod[:,:self.hidden[3],:,:,:]
        self.list_skip_flow[3]['logvar']['Flair'] = flow_mod[:,self.hidden[3]:,:,:,:]

        flow_mod = images['T1c']
        flow_mod = self.cnn_T1c_0(flow_mod)
        flow_mod = self.cnn_T1c_1(flow_mod)
        self.list_skip_flow[0]['mu']['T1c'] = flow_mod[:,:self.hidden[0],:,:,:]
        self.list_skip_flow[0]['logvar']['T1c'] = flow_mod[:,self.hidden[0]:,:,:,:]
        flow_mod = self.cnn_T1c_2(flow_mod)
        flow_mod = self.cnn_T1c_3(flow_mod)
        self.list_skip_flow[1]['mu']['T1c'] = flow_mod[:,:self.hidden[1],:,:,:]
        self.list_skip_flow[1]['logvar']['T1c'] = flow_mod[:,self.hidden[1]:,:,:,:]
        flow_mod = self.cnn_T1c_4(flow_mod)
        flow_mod = self.cnn_T1c_5(flow_mod)
        self.list_skip_flow[2]['mu']['T1c'] = flow_mod[:,:self.hidden[2],:,:,:]
        self.list_skip_flow[2]['logvar']['T1c'] = flow_mod[:,self.hidden[2]:,:,:,:]
        flow_mod = self.cnn_T1c_6(flow_mod)
        flow_mod = self.cnn_T1c_7(flow_mod)
        self.list_skip_flow[3]['mu']['T1c'] = flow_mod[:,:self.hidden[3],:,:,:]
        self.list_skip_flow[3]['logvar']['T1c'] = flow_mod[:,self.hidden[3]:,:,:,:]

        flow_mod = images['T1']
        flow_mod = self.cnn_T1_0(flow_mod)
        flow_mod = self.cnn_T1_1(flow_mod)
        self.list_skip_flow[0]['mu']['T1'] = flow_mod[:,:self.hidden[0],:,:,:]
        self.list_skip_flow[0]['logvar']['T1'] = flow_mod[:,self.hidden[0]:,:,:,:]
        flow_mod = self.cnn_T1_2(flow_mod)
        flow_mod = self.cnn_T1_3(flow_mod)
        self.list_skip_flow[1]['mu']['T1'] = flow_mod[:,:self.hidden[1],:,:,:]
        self.list_skip_flow[1]['logvar']['T1'] = flow_mod[:,self.hidden[1]:,:,:,:]
        flow_mod = self.cnn_T1_4(flow_mod)
        flow_mod = self.cnn_T1_5(flow_mod)
        self.list_skip_flow[2]['mu']['T1'] = flow_mod[:,:self.hidden[2],:,:,:]
        self.list_skip_flow[2]['logvar']['T1'] = flow_mod[:,self.hidden[2]:,:,:,:]
        flow_mod = self.cnn_T1_6(flow_mod)
        flow_mod = self.cnn_T1_7(flow_mod)
        self.list_skip_flow[3]['mu']['T1'] = flow_mod[:,:self.hidden[3],:,:,:]
        self.list_skip_flow[3]['logvar']['T1'] = flow_mod[:,self.hidden[3]:,:,:,:]

        flow_mod = images['T2']
        flow_mod = self.cnn_T2_0(flow_mod)
        flow_mod = self.cnn_T2_1(flow_mod)
        self.list_skip_flow[0]['mu']['T2'] = flow_mod[:,:self.hidden[0],:,:,:]
        self.list_skip_flow[0]['logvar']['T2'] = flow_mod[:,self.hidden[0]:,:,:,:]
        flow_mod = self.cnn_T2_2(flow_mod)
        flow_mod = self.cnn_T2_3(flow_mod)
        self.list_skip_flow[1]['mu']['T2'] = flow_mod[:,:self.hidden[1],:,:,:]
        self.list_skip_flow[1]['logvar']['T2'] = flow_mod[:,self.hidden[1]:,:,:,:]
        flow_mod = self.cnn_T2_4(flow_mod)
        flow_mod = self.cnn_T2_5(flow_mod)
        self.list_skip_flow[2]['mu']['T2'] = flow_mod[:,:self.hidden[2],:,:,:]
        self.list_skip_flow[2]['logvar']['T2'] = flow_mod[:,self.hidden[2]:,:,:,:]
        flow_mod = self.cnn_T2_6(flow_mod)
        flow_mod = self.cnn_T2_7(flow_mod)
        self.list_skip_flow[3]['mu']['T2'] = flow_mod[:,:self.hidden[3],:,:,:]
        self.list_skip_flow[3]['logvar']['T2'] = flow_mod[:,self.hidden[3]:,:,:,:]

        return self.list_skip_flow

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = ConvEncoder()
        self.approximate_sampler = GaussianSampler()
        self.img_decoder = ConvDecoderImg()
        self.mod_img = MODALITIES[:4]
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight) 

    def forward(self, images, choices,is_inference=False):
        '''
        images{
            'Flair':[B,1,112,112,112]
            'T1c':[B,1,112,112,112]
            'T1':[B,1,112,112,112]
            'T2':[B,1,112,112,112]
        }
        '''
        post_param = self.encoder(images)
        '''
        post_param{
id=0        {mu:{Flair:[B,4,112,112,112],T1c:[B,4,112,112,112],T1:[B,4,112,112,112],T2:[B,4,112,112,112]},
             logvar:{Flair:[B,4,112,112,112],T1c:[B,4,112,112,112],T1:[B,4,112,112,112],T2:[B,4,112,112,112]}},

id=1        {mu:{Flair:[B,8,56,56,56],T1c:[B,8,56,56,56],T1:[B,8,56,56,56],T2:[B,8,56,56,56]},
             logvar:{Flair:[B,8,56,56,56],T1c:[B,8,56,56,56],T1:[B,8,56,56,56],T2:[B,8,56,56,56]}},

id=2        {mu:{Flair:[B,16,28,28,28],T1c:[B,16,28,28,28],T1:[B,16,28,28,28],T2:[B,16,28,28,28]},
             logvar:{Flair:[B,16,28,28,28],T1c:[B,16,28,28,28],T1:[B,16,28,28,28],T2:[B,16,28,28,28]}},

id=3        {mu:{Flair:[B,32,14,14,14],T1c:[B,32,14,14,14],T1:[B,32,14,14,14],T2:[B,32,14,14,14]},
             logvar:{Flair:[B,32,14,14,14],T1c:[B,32,14,14,14],T1:[B,32,14,14,14],T2:[B,32,14,14,14]}}             
        }
        '''
        skip_flow = []
        for k in range(len(post_param)):        #range(4)
            sample = self.approximate_sampler(post_param[k]['mu'], post_param[k]['logvar'],self.mod_img,choices,is_inference)
                                                #    means               logvar               list_mod  choices is_inference
            skip_flow.append(sample)#{[B,4,112,112,112],[B,8,56,56,56],[B,16,28,28,28],[B,32,14,14,14]}
        img_output = self.img_decoder(skip_flow,is_inference)
        #[Flair:[B,1,112,112,112],T1c:[B,1,112,112,112],T1:[B,1,112,112,112],T2:[B,1,112,112,112]]
        if is_inference:
            return img_output['seg']
        else:
            return img_output, post_param