import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelBlock(nn.Module):
    def __init__(self, in_filte,out_filte):
        super(ChannelBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filte, out_filte, kernel_size=(3, 3), padding=3, dilation=3)
        self.batch1 = nn.BatchNorm2d(out_filte)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filte, out_filte, kernel_size=(5, 5), padding=2)
        self.batch2 = nn.BatchNorm2d(out_filte)
        self.relu2 = nn.LeakyReLU()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2*out_filte, out_filte)
        self.batch3 = nn.BatchNorm1d(out_filte)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(out_filte, out_filte)
        self.sigmoid = nn.Sigmoid()

        self.conv3 = nn.Conv2d(out_filte * 2, out_filte, kernel_size=1, padding=0)
        self.batch4 = nn.BatchNorm2d(out_filte)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        batch1 = self.batch1(conv1)
        leakyReLU1 = self.relu1(batch1)

        conv2 = self.conv2(x)
        batch2 = self.batch2(conv2)
        leakyReLU2 = self.relu2(batch2)

        data3 = torch.cat([leakyReLU1, leakyReLU2], dim=1)
        data3 = self.global_pool(data3)
        data3 = data3.view(data3.size(0), -1)
        data3 = self.fc1(data3)
        data3 = self.batch3(data3)
        data3 = self.relu3(data3)
        data3 = self.fc2(data3)
        data3 = self.sigmoid(data3)

        a = data3.view(data3.size(0), -1, 1, 1)

        a1 = 1 - data3
        a1 = a1.view(a1.size(0), -1, 1, 1)

        y = leakyReLU1 * a

        y1 = leakyReLU2 * a1

        data_a_a1 = torch.cat([y, y1], dim=1)

        conv3 = self.conv3(data_a_a1)
        batch3 = self.batch4(conv3)
        leakyReLU3 = self.relu4(batch3)

        return leakyReLU3


class SpatialBlock(nn.Module):
    def __init__(self, in_filte,out_filte):
        super(SpatialBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filte, out_filte, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(out_filte)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_filte, out_filte, kernel_size=1, padding=0)
        self.batch2 = nn.BatchNorm2d(out_filte)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(2 * out_filte, out_filte, kernel_size=1, padding=0)
        self.batch3 = nn.BatchNorm2d(out_filte)

        self.conv4 = nn.Conv2d(out_filte, 1, kernel_size=1, padding=0)

    def forward(self, data, channel_data):
        conv1 = self.conv1(data)
        batch1 = self.batch1(conv1)
        relu1 = self.relu1(batch1)

        conv2 = self.conv2(relu1)
        batch2 = self.batch2(conv2)
        FS1 = self.relu2(batch2)

        data3 = channel_data + FS1
        data3 = F.relu(data3)
        data3 = self.conv4(data3)
        data3 = torch.sigmoid(data3)

        a = data3.expand_as(channel_data)
        y = a * channel_data


        a1 = 1 - data3
        a1 = a1.expand_as(FS1)
        y1 = a1 * FS1

        data_a_a1 = torch.cat((y, y1), dim=1)

        conv3 = self.conv3(data_a_a1)
        batch3 = self.batch3(conv3)

        return batch3

class up(nn.Module):
    def __init__(self):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2)
        )
    def forward(self, x):
        x = self.up(x)
        return x

class AAU(nn.Module):
    def __init__(self, in_channel=1, out_channel=2):
        super(AAU, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.channel1 = ChannelBlock(in_channel,32)
        self.spatial1 = SpatialBlock(in_channel,32)

        self.channel2 = ChannelBlock(32, 32)
        self.spatial2 = SpatialBlock(32, 32)

        self.channel3 = ChannelBlock(32, 64)
        self.spatial3 = SpatialBlock(32, 64)

        self.channel4 = ChannelBlock(64, 64)
        self.spatial4 = SpatialBlock(64, 64)

        self.channel5 = ChannelBlock(64, 128)
        self.spatial5 = SpatialBlock(64, 128)

        self.channel6 = ChannelBlock(128, 128)
        self.spatial6 = SpatialBlock(128, 128)

        self.channel7 = ChannelBlock(128, 256)
        self.spatial7 = SpatialBlock(128, 256)

        # self.channel8 = ChannelBlock(256, 256)
        # self.spatial8 = SpatialBlock(256, 256)

        #self.channel9 = ChannelBlock(256, 512)
        #self.spatial9 = SpatialBlock(256, 512)

        #self.channel10 = ChannelBlock(512, 512)
        #self.spatial10 = SpatialBlock(512, 512)

        self.upsample = up()

        #self.dchannel1 = ChannelBlock(512, 256)
        #self.dspatial1 = SpatialBlock(512, 256)

        #self.dchannel2 = ChannelBlock(256, 256)
        #self.dspatial2 = SpatialBlock(256, 256)

        self.dchannel1 = ChannelBlock(256, 128)
        self.dspatial1 = SpatialBlock(256, 128)

        self.dchannel2 = ChannelBlock(128, 128)
        self.dspatial2 = SpatialBlock(128, 128)

        self.dchannel3 = ChannelBlock(128, 64)
        self.dspatial3 = SpatialBlock(128, 64)

        self.dchannel4 = ChannelBlock(64, 64)
        self.dspatial4 = SpatialBlock(64, 64)

        self.dchannel5 = ChannelBlock(64, 32)
        self.dspatial5 = SpatialBlock(64, 32)

        self.dchannel6 = ChannelBlock(32, 32)
        self.dspatial6 = SpatialBlock(32, 32)

        self.conv_1x1 = nn.Conv2d(32, out_channel, kernel_size=1, stride=1, padding=0)
        self.batch1=nn.BatchNorm2d(out_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_data1 = self.channel1(x)
        spatial_data1 = self.spatial1(x ,channel_data1)

        # channel_data2 = self.channel2(spatial_data1)
        # spatial_data2 = self.spatial2(spatial_data1,channel_data2)

        x1 = self.Maxpool(spatial_data1)

        channel_data3 = self.channel3(x1)
        spatial_data3 = self.spatial3(x1,channel_data3)

        # channel_data4 = self.channel4(spatial_data3)
        # spatial_data4 = self.spatial4(spatial_data3,channel_data4)

        x2 = self.Maxpool(spatial_data3)

        channel_data5 = self.channel5(x2)
        spatial_data5 = self.spatial5(x2,channel_data5)

        # channel_data6 = self.channel6(spatial_data5)
        # spatial_data6 = self.spatial6(spatial_data5,channel_data6)

        x3 = self.Maxpool(spatial_data5)

        channel_data7 = self.channel7(x3)
        spatial_data7 = self.spatial7(x3, channel_data7)

        # channel_data8 = self.channel8(spatial_data7)
        # spatial_data8 = self.spatial8(spatial_data7, channel_data8)

        x4 = self.upsample(spatial_data7)

        channel_data_d1 = self.dchannel1(x4)
        spatial_data_d1 = self.dspatial1(x4,channel_data_d1)

        # channel_data_d2 = self.dchannel2(spatial_data_d1)
        # spatial_data_d2 = self.dspatial2(spatial_data_d1,channel_data_d2)

        x5 = self.upsample(spatial_data_d1)

        channel_data_d3 = self.dchannel3(x5)
        spatial_data_d3 = self.dspatial3(x5, channel_data_d3)

        # channel_data_d4 = self.dchannel4(spatial_data_d3)
        # spatial_data_d4 = self.dspatial4(spatial_data_d3, channel_data_d4)

        x6 = self.upsample(spatial_data_d3)

        channel_data_d5 = self.dchannel5(x6)
        spatial_data_d5 = self.dspatial5(x6, channel_data_d5)

        # channel_data_d6 = self.dchannel6(spatial_data_d5)
        # spatial_data_d6 = self.dspatial6(spatial_data_d5, channel_data_d6)

        out = self.conv_1x1(spatial_data_d5)
        out = self.batch1(out)
        out = self.sigmoid(out)
        out = F.softmax(out,dim=1)  # mine

        return out

if __name__ == '__main__':
    # test network forward

    net = AAU(1,2)
    #print(net)
    in1 = torch.randn((2,1,64,64))
    out1 = net(in1)
    print("结果")
    print(out1.size())