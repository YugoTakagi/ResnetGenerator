import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np


# @TODO AdaptedToPatchNCE
class ResnetGenerator_9blocks_2to1(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_bias=True):
        super(ResnetGenerator_9blocks_2to1, self).__init__()
        self.encoder = MyEncoder(input_nc=input_nc, output_nc=output_nc)
        self.decoder = MyDecoder(input_nc=input_nc, output_nc=output_nc)

    def forward(self, x, layers=[0, 4, 8, 12, 16], encode_only=False):
        if (encode_only):
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.encoder.model):
                a = 0
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                else:
                    pass
                if layer_id == layers[-1]:
                    return feats
        else: 
            z = self.encoder(x)
            y = self.decoder(z)

        return y


class MyEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_bias=True):
        super(MyEncoder, self).__init__()

        model = [
            # (3, 256, 256)
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=input_nc, out_channels=ngf, kernel_size=7, padding=0, bias=use_bias),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),

            # (64, 255, 255)

            # 本当はDownsampleする．-----
            nn.Conv2d(in_channels=ngf, out_channels=ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            Downsample(ngf * 2),

            nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 2 * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.InstanceNorm2d(ngf * 2 * 2),
            nn.ReLU(True),
            Downsample(ngf * 2 * 2),

            # (256, 64, 64)

            # Resnet -----
            MyResnetBlock(dim=ngf * 2 * 2, use_bias=use_bias),
            MyResnetBlock(dim=ngf * 2 * 2, use_bias=use_bias),
            MyResnetBlock(dim=ngf * 2 * 2, use_bias=use_bias),
            MyResnetBlock(dim=ngf * 2 * 2, use_bias=use_bias),
            MyResnetBlock(dim=ngf * 2 * 2, use_bias=use_bias)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        z = self.model(x)
        return z


class MyDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_bias=True):
        super(MyDecoder, self).__init__()
        model = [
            # encoderの続き．
            # (256, 64, 64)
            MyResnetBlock(dim=ngf * 2 * 2, use_bias=use_bias),
            MyResnetBlock(dim=ngf * 2 * 2, use_bias=use_bias),
            MyResnetBlock(dim=ngf * 2 * 2, use_bias=use_bias),
            MyResnetBlock(dim=ngf * 2 * 2, use_bias=use_bias),

            # (255, 64, 64)

            # アップサンプリング．
            Upsample(channels=ngf * 2 * 2),
            nn.Conv2d(ngf * 2 * 2, ngf * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),

            Upsample(channels=ngf * 2),
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),

            # (128, 256, 256)

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class MyResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(MyResnetBlock, self).__init__()

        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        y = x + self.model(x)
        return y



# 以下，CUTに感謝．
class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        # self.pad = get_pad_layer(pad_type)(self.pad_sizes)
        self.pad = nn.ReflectionPad2d(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        # self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])
        self.pad = nn.ReplicationPad2d([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


if __name__ == '__main__':
    model = ResnetGenerator_9blocks_2to1(input_nc=3, output_nc=3)
    summary(model, (3, 255, 255), batch_size=1, device="cpu") # summaryはバッチサイズ2を必ず入力してくる.
    
    # model = ResnetGenerator_9blocks_2to1(input_nc=6, output_nc=3) 
    # summary(model, (6, 255, 255), batch_size=1, device="cpu") # summaryはバッチサイズ2を必ず入力してくる.