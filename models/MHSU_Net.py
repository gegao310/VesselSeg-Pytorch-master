from functools import reduce
from this import d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
#from .Transformer import *
#from timm.models.vision_transformer import VisionTransformer, _cfg


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
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



class mcb1(nn.Module):

    def __init__(self, in_channels,out_channels):
        super(mcb1, self).__init__()

        self.branch1 = nn.Sequential(
            conv_block_3(in_channels,out_channels),
        )

        self.branch2 = nn.Sequential(
            conv_block_3(in_channels,out_channels),
            conv_block_3(out_channels,out_channels),
        )

        self.branch3 = nn.Sequential(
            conv_block_3(in_channels,out_channels),
            conv_block_3(out_channels,out_channels),
            conv_block_3(out_channels,out_channels),
        )

        self.conv1x1 = nn.Conv2d(3*out_channels,out_channels,kernel_size=1,stride=1,padding=0)


    def forward(self, x):

        outputs1 = self.branch1(x)
        outputs2 = self.branch2(x)
        outputs3 = self.branch3(x)

        outputs4 = torch.cat((outputs1, outputs2, outputs3), dim=1)
        outputs = self.conv1x1(outputs4)
        return outputs

class mcb(nn.Module):

    def __init__(self, in_channels,out_channels):
        super(mcb, self).__init__()

        self.branch1 = nn.Sequential(
            conv_block_1(in_channels,in_channels//2),
            conv_block_3(in_channels // 2,out_channels),
        )

        self.branch2 = nn.Sequential(
            conv_block_1(in_channels,in_channels//2),
            conv_block_3(in_channels//2,out_channels),
            conv_block_3(out_channels,out_channels),
        )

        self.branch3 = nn.Sequential(
            conv_block_1(in_channels,in_channels//2),
            conv_block_3(in_channels//2,out_channels),
            conv_block_3(out_channels,out_channels),
            conv_block_3(out_channels,out_channels),
        )

        self.conv1x1 = nn.Conv2d(3*out_channels,out_channels,kernel_size=1,stride=1,padding=0)


    def forward(self, x):

        outputs1 = self.branch1(x)
        outputs2 = self.branch2(x)
        outputs3 = self.branch3(x)

        outputs4 = torch.cat((outputs1, outputs2, outputs3), dim=1)
        outputs = self.conv1x1(outputs4)
        return outputs



class hdsb(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(hdsb, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_3x3 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1)
        self.se = SELayer(channel=3 * out_channels)
        self.conv_1 = nn.Conv2d(out_channels*3,out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self,x):

        branch1 = self.Maxpool(x)
        branch2 = self.Avgpool(x)
        branch3 = self.conv_3x3(x)

        concat = torch.cat((branch1, branch2, branch3),1)
        se = self.se(concat)
        outputs = self.conv_1(se)
        return outputs



class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class CIF_block(nn.Module):
    def __init__(self,in_channels,out_channels,r=16,b=3):
        super(CIF_block,self).__init__()
        d = in_channels//r
        self.b = b
        self.out_channels = out_channels
        self.atrous_conv = nn.ModuleList()
        for i in range(b):
            self.atrous_conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,d,kernel_size=3,stride=1,padding=i+1,dilation=i+1),
                    nn.BatchNorm2d(d),
                    nn.ReLU(inplace=True)
                )
            )
        self.gb = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Sequential(   
            nn.Conv2d(d,d,1,bias=False),     
            #nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2=nn.Conv2d(d,d*3,1,1,bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        outputs1 = []     
        for i,atrous_conv in enumerate(self.atrous_conv):
            outputs1.append(atrous_conv(x))
        F = reduce(lambda x,y:x+y, outputs1)  #二元操作函数内逐元素相加(利用lambda内嵌一个函数的定义)
        S = self.gb(F)
        Z = self.fc1(S)
        abc = self.fc2(Z)  #升维
        abc = torch.reshape(abc, (batch_size, self.b,self.out_channels//16, -1))  #调整形状，变为三个全连接层的值
        abc = self.softmax(abc)   #使得三个全连接层的对应位置进行softmax
        abc = list(abc.chunk(self.b,dim=1)) #pytorch方法chunk用来分片，按照指定维度分成3个tensor块
        abc = list(map(lambda x: torch.reshape(x, (batch_size, self.out_channels // 16, 1, 1)), abc))  #将所有分块调整形状

        outputs2 = list(map(lambda x,y:x*y,outputs1,abc))  #三个权重和对应原始输入相乘
        outputs = reduce(lambda x,y:x+y, outputs2)   #三个相乘的结果再逐元素相加
        return outputs

class Skip_Connection_Plus(nn.Module):
    def __init__(self,in_channels,out_channels, num):
        super(Skip_Connection_Plus,self).__init__()
        self.CIF_block =  CIF_block(in_channels, out_channels)
        self.con1 = conv_block_1(out_channels + num, in_channels)  #cat之后维度变化，不能直接取in_channels

    def forward(self,x):
        CIF_block = self.CIF_block(x)
        concatenate1 = torch.cat((CIF_block, x),1)
        outputs = self.con1(concatenate1)
        return outputs

class MHSU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=2,t=2):
        super(MHSU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.mcb1 = mcb1(in_channels=img_ch, out_channels=64)
        self.hdsb1 = hdsb(in_channels=64, out_channels=64)
        self.mcb2 = mcb(in_channels=64, out_channels=128)
        self.hdsb2 = hdsb(in_channels=128, out_channels=128)
        self.mcb3 = mcb(in_channels=128, out_channels=256)
        self.hdsb3 = hdsb(in_channels=256, out_channels=256)
        self.mcb4 = mcb(in_channels=256, out_channels=512)
        self.hdsb4 = hdsb(in_channels=512, out_channels=512)
        self.mcb5 = mcb(in_channels=512, out_channels=1024)

        self.Up5 = up_conv(in_channels=1024, out_channels=512)
        self.skip_c5 = Skip_Connection_Plus(in_channels=512, out_channels=512,num=32)
        self.Up_conv5 = conv_block(in_channels=1024, out_channels=512)
         

        self.Up4 = up_conv(in_channels=512, out_channels=256)
        # self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.skip_c4 = Skip_Connection_Plus(in_channels=256, out_channels=256, num=16)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(in_channels=256, out_channels=128)
        # self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.skip_c3 = Skip_Connection_Plus(in_channels=128, out_channels=128, num=8)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(in_channels=128, out_channels=64)
        # self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.skip_c2 = Skip_Connection_Plus(in_channels=64, out_channels=64, num=4)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x1 = self.mcb1(x)
        x2 = self.hdsb1(x1)
        x2 = self.mcb2(x2)
        x3 = self.hdsb2(x2)
        x3 = self.mcb3(x3)
        x4 = self.hdsb3(x3)
        x4 = self.mcb4(x4)
        x5 = self.hdsb4(x4)
        x5 = self.mcb5(x5)

        x4 = self.skip_c5(x4)
        x3 = self.skip_c4(x3)
        x2 = self.skip_c3(x2)
        x1 = self.skip_c2(x1)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)    
        d5 = self.Up_conv5(d5)            

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)    
        d4 = self.Up_conv4(d4) 

        d3 = self.Up3(x3)
        d3 = torch.cat((x2, d3), dim=1)    
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(x2)
        d2 = torch.cat((x1, d2), dim=1)    
        d2 = self.Up_conv2(d2)  

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)  # mine

        return d1

