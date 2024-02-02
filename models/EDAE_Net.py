import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from torch.nn import init
from torch.autograd import Variable
from torchvision import transforms as T


# Corresponding relationship of network and modules:
# transfer-->MDAE
# EdgeEnhance+MRF-->CPSE
# MFB-->DGF
# scalstt-->AWL


nonlinearity = partial(F.relu, inplace=True)

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class BN_Conv2d(nn.Module):
    """
    BN_CONV_RELU
    """
    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv2d, self).__init__()
        
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
            
        )
        

    def forward(self, x):
        
        x1  =self.seq(x)

        #return  F.relu(x1)
        return  x1

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class bn(nn.Module):
    def __init__(self,ch_out):
        super(bn,self).__init__()
        self.bn = nn.Sequential(
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.bn(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, groups=group, bias=bias)

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class Residual_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Residual_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0,bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )  

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv(x1)
        x = x1+x2
        #x = nn.ReLU(inplace=True)
        return x 


class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_dim, k_size=3):
        super(ECALayer, self).__init__()
        self.channel_in = in_dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_dim, k_size=3):
        super(ECALayer, self).__init__()
        self.channel_in = in_dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)



class EdgeEnhance(nn.Module):

    def __init__(self,):
        super(EdgeEnhance, self).__init__()
        self.Maxpool=nn.MaxPool1d(kernel_size=3,stride=1,padding=1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        x_W=x.view(m_batchsize, -1, C*height).permute(0,2,1)
        #pool for W
        x_W = self.Maxpool(x_W)
        x_W =x_W.view(m_batchsize, C, height, width)
        #pool for H
        x_H=x.view(m_batchsize, -1, C*width).permute(0,2,1)
        x_H = self.Maxpool(x_H)

        x_W =x_W.reshape(m_batchsize, C, height, width)
        x_H =x_H.reshape(m_batchsize, C, height, width)
        x = torch.max(x_W,x_H)

        return x

class MRF(nn.Module):

    def __init__(self,):
        super(MRF, self).__init__()
        self.conv128=nn.Conv2d(512, 64, kernel_size=1,stride=1,padding=0,bias=True)
        self.conv512=nn.Conv2d(64, 512, kernel_size=1,stride=1,padding=0,bias=True)
        self.dilate1=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1,dilation=1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dilate3=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=3,dilation=3,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dilate5=nn.Sequential(
            nn.Conv2d(64,64, kernel_size=3,stride=1,padding=5,dilation=5,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x= self.conv128(x)
        x1=self.dilate1(x)
        x = x + x1
        x2=self.dilate3(x)
        x = x + x2
        x3=self.dilate5(x)
        # print(x3.shape)
        x = x + x3
        x = self.conv512(x)
    
        return x
        

class transfer(nn.Module):

    def __init__(self,F_C,F_H,F_W):
        super(transfer, self).__init__()

        self.EAC_C = ECALayer(in_dim=F_C)
        self.EAC_H = ECALayer(in_dim=F_H)
        self.EAC_W = ECALayer(in_dim=F_W)

        self.EH =EdgeEnhance()

    def forward(self, x):
        # batch, c, h, w

        # calculate c attention        
        x_out11=self.EAC_C(x)

        # calculate h attention
        x_perm2 = x.permute(0,2,1,3).contiguous()
        x_out21 = self.EAC_W(x_perm2)   
        x_out21 = x_out21.permute(0,2,1,3).contiguous()

        # calculate w attention
        x_perm3 = x.permute(0,3,2,1).contiguous()
        x_out31 = self.EAC_H(x_perm3)      
        x_out31 = x_out31.permute(0,3,2,1).contiguous()

    
        x_out = (1 / 3) * (x_out11 + x_out21 + x_out31)
    
        return x_out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=12, pool_types=['avg', 'max']):

    #def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        #print(x.shape)
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
                #print("avg",channel_att_raw.shape)            
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
                #print("max",channel_att_raw.shape)  
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
                #print("lp",channel_att_raw.shape)  
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
                #print("lse",channel_att_raw.shape)  

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
            
            #print(channel_att_sum.shape)


        # scalecoe = F.sigmoid(channel_att_sum)
        #print(channel_att_sum.shape[0])
        channel_att_sum = channel_att_sum.reshape(channel_att_sum.shape[0], 3, 4)
        avg_weight = torch.mean(channel_att_sum, dim=2).unsqueeze(2)
        avg_weight = avg_weight.expand(channel_att_sum.shape[0], 3, 4).reshape(channel_att_sum.shape[0], 12)
        scale = F.sigmoid(avg_weight).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale, scale

class SpatialAtten(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, stride=1):
        super(SpatialAtten, self).__init__()
        self.conv1 = BasicConv(in_size, out_size, kernel_size, stride=stride,
                               padding=(kernel_size-1) // 2, relu=True)
        self.conv2 = BasicConv(out_size, out_size, kernel_size=1, stride=stride,
                               padding=0, relu=True, bn=False)

    def forward(self, x):
        residual = x
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        spatial_att = F.sigmoid(x_out).unsqueeze(4).permute(0, 1, 4, 2, 3)
        spatial_att = spatial_att.expand(spatial_att.shape[0], 3, 4, spatial_att.shape[3], spatial_att.shape[4]).reshape(
                                        spatial_att.shape[0], 12, spatial_att.shape[3], spatial_att.shape[4])
        x_out = residual * spatial_att

        x_out += residual

        return x_out, spatial_att

class Scale_atten_block(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=12, pool_types=['avg', 'max'], no_spatial=False):
    #def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(Scale_atten_block, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialAtten(gate_channels, gate_channels //reduction_ratio)

    def forward(self, x):
        #print(x.shape)
        x_out, ca_atten = self.ChannelGate(x)
        if not self.no_spatial:
            x_out, sa_atten = self.SpatialGate(x_out)

        return x_out, ca_atten, sa_atten

class scale_atten_convblock(nn.Module):
    def __init__(self, in_size, out_size, stride=1, downsample=None, use_cbam=True, no_spatial=False, drop_out=False):
        super(scale_atten_convblock, self).__init__()

        self.downsample = downsample
        self.stride = stride
        self.no_spatial = no_spatial
        self.dropout = drop_out

        self.relu = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(in_size, out_size)
        self.bn3 = nn.BatchNorm2d(out_size)

        if use_cbam:
            self.cbam = Scale_atten_block(in_size, reduction_ratio=4, no_spatial=self.no_spatial)  # out_size
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out, scale_c_atten, scale_s_atten = self.cbam(x)


        out += residual
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


class MFB(nn.Module):
    def __init__(self,ch_in,ch_out,T_size):
        super(MFB,self).__init__()
        self.size=T_size
        self.conv1=nn.Conv2d(ch_in[-1], 64, 3, padding=1, bias=False)
        self.bn_1=bn(64)
        self.conv2=nn.Conv2d(ch_in[-2], 64, 3, padding=1, bias=False)
        self.bn_2=bn(64)   
        self.ECA=ECALayer(128)
        self.Conv_1x1 = nn.Conv2d(128, ch_out, kernel_size=1, stride=1, padding=0)
        self.relu =nn.ReLU(inplace=True)

    def forward(self,*inputs):
        t = T.Resize([self.size,self.size])
        x1=inputs[-1]
        x1=t(x1)
        x1=self.conv1(x1)
        x1=self.bn_1(x1)
        x2 = inputs[-2]
        x3=self.conv2(x2)
        x3=self.bn_2(x3)
        x=torch.cat((x1, x3), dim=1)
        x=self.ECA(x)
        x=self.Conv_1x1(x)
        x=x*x2
        return x


class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.scalstt=scale_atten_convblock(12,4)
        self.upscale_8=nn.Upsample(scale_factor=8)
        self.upscale_4=nn.Upsample(scale_factor=4)
        self.upscale_2=nn.Upsample(scale_factor=2)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.trans1=transfer(F_C=64,F_H=64,F_W=64)
        self.trans2=transfer(F_C=128,F_H=32,F_W=32)
        self.trans3=transfer(F_C=256,F_H=16,F_W=16)

        self.mfb_1=MFB([256,512],ch_out=256,T_size=16)
        self.mfb_2=MFB([128,256],ch_out=128,T_size=32)
        self.mfb_3=MFB([64,128],ch_out=64,T_size=64)

        self.Edge = EdgeEnhance()
        self.MRF =MRF()

        self.Conv_1x1_256 = nn.Conv2d(256, 4, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_128 = nn.Conv2d(128, 4, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_64 = nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0)       
        self.Conv_1x1 = nn.Conv2d(4, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x1 = self.trans1(x1)
        x2 = self.trans2(x2)
        x3 = self.trans3(x3)

        x4 = self.Edge(x4) 
        x4 = self.MRF(x4)
        # x4 = self.trans4(x4)

        m1 = self.mfb_1(x3,x4)
        m2 = self.mfb_2(x2,x3)
        m3 = self.mfb_3(x1,x2)

        d4 = self.Up4(x4)
        d4 = torch.cat((m1, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((m2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((m3, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d4 = self.Conv_1x1_256(d4)
        d4 = self.upscale_4(d4)
        d3 = self.Conv_1x1_128(d3)
        d3 = self.upscale_2(d3)
        d2 = self.Conv_1x1_64(d2)

        d2 = torch.cat((d2, d3, d4), dim=1)
        d2 = self.scalstt(d2)


        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)
        return d1