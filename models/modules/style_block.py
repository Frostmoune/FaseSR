import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable
from . import block as B

from math import sqrt

import random

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super(NoiseInjection, self).__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super(AdaptiveInstanceNorm, self).__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input_, style):
        out = self.norm(input_)
        if style is None:
            return out

        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = gamma * out + beta

        return out

class RRDB_Unet(nn.Module):

    def __init__(self, nc, style_dim=512, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB_Unet, self).__init__()
        self.RDB1 = B.ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = B.ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = B.ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)

        unet_block = UnetBlock(in_nc=None, mid_nc=nc * 4, out_nc=nc * 4, submodule=None, innermost=True, up_kernel=(3, 4))
        unet_block = UnetBlock(in_nc=None, mid_nc=nc * 4, out_nc=nc * 2, submodule=unet_block, down_kernel=(2, 3), up_kernel=(4, 3))
        unet_block = UnetBlock(in_nc=None, mid_nc=nc * 2, out_nc=nc, submodule=unet_block)
        self.Unet = UnetBlock(in_nc=nc, mid_nc=nc, out_nc=nc, submodule=unet_block, outermost=True)
        self.conv = B.convBlock(nc * 2, nc, kernel_size=3, norm_type=None, act_type=act_type)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out = torch.cat((out, self.Unet(x)), 1)
        out = self.conv(out)
        out = out.mul(0.2) + x
        return out

# Add RRDB_AdaIn
class RRDB_AdaIn(nn.Module):

    def __init__(self, nc, style_dim=512, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(RRDB_AdaIn, self).__init__()
        self.RDB1 = B.ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB2 = B.ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.RDB3 = B.ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, act_type, mode)
        self.AdaIn = AdaptiveInstanceNorm(nc, style_dim)

    def forward(self, x, style):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out = out.mul(0.2) + x
        return self.AdaIn(out, style)

class StyledconvBlock(nn.Module):

    def __init__(self, in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, 
                bias=True, act_type='relu', with_style=True, style_dim=512, with_noise=False):
        super(StyledconvBlock, self).__init__()

        padding = get_valid_padding(kernel_size, dilation)
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)

        if with_noise:
            self.noise = NoiseInjection(out_nc)
        else:
            self.noise = None

        if with_style:
            self.adain = AdaptiveInstanceNorm(out_nc, style_dim)
        else:
            self.adain = None

        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'leakyrelu':
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = None
    
    def forward(self, input_, style):
        out = self.conv(input_)
        if not (self.noise is None):
            batch, _, W, H = out.shape
            noise = torch.randn(batch, 1, W, H).to(out[0].device)
            out = self.noise(out, noise)
        
        if not (self.adain is None):
            out = self.adain(out, style)

        if not (self.act is None):
            out = self.act(out)

        return out

class StyledResNetBlock(nn.Module):
    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, act_type='relu', res_scale=1, with_style=True, style_dim=512, with_noise=False):
        super(StyledResNetBlock, self).__init__()
        conv0 = StyledconvBlock(in_nc, mid_nc, kernel_size, stride=stride, dilation=dilation, 
                    bias=bias, act_type=act_type, with_style=with_style, style_dim=style_dim, with_noise=with_noise)
        conv1 = StyledconvBlock(mid_nc, out_nc, kernel_size, stride=stride, dilation=dilation, 
                    bias=bias, act_type=act_type, with_style=with_style, style_dim=style_dim, with_noise=with_noise)
        self.res = nn.ModuleList([conv0, conv1])
        self.res_scale = res_scale

    def forward(self, x, style):
        res = x
        for block in self.res:
            res = block(res, style)
        res = res.mul(self.res_scale)
        return x + res

class StyledResidualDenseBlock(nn.Module):
    
    def __init__(self, nc, kernel_size=3, gc=32, stride=1, dilation=1, bias=True, 
                act_type='leakyrelu', with_style=True, style_dim=512, with_noise=False):
        super(StyledResidualDenseBlock, self).__init__()
        self.conv1 = StyledconvBlock(nc, gc, kernel_size, stride=stride, dilation=dilation, 
                        bias=bias, act_type=act_type, with_style=with_style, style_dim=style_dim, with_noise=with_noise)
        self.conv2 = StyledconvBlock(nc+gc, gc, kernel_size, stride=stride, dilation=dilation, 
                        bias=bias, act_type=act_type, with_style=with_style, style_dim=style_dim, with_noise=with_noise)
        self.conv3 = StyledconvBlock(nc+gc*2, gc, kernel_size, stride=stride, dilation=dilation, 
                        bias=bias, act_type=act_type, with_style=with_style, style_dim=style_dim, with_noise=with_noise)
        self.conv4 = StyledconvBlock(nc+gc*3, gc, kernel_size, stride=stride, dilation=dilation, 
                        bias=bias, act_type=act_type, with_style=with_style, style_dim=style_dim, with_noise=with_noise)
        self.conv5 = StyledconvBlock(nc+gc*4, nc, kernel_size, stride=stride, dilation=dilation, 
                        bias=bias, act_type=None, with_style=with_style, style_dim=style_dim, with_noise=with_noise)
    
    def forward(self, x, style):
        x1 = self.conv1(x, style)
        x2 = self.conv2(torch.cat((x, x1), 1), style)
        x3 = self.conv3(torch.cat((x, x1, x2), 1), style)
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1), style)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1), style)
        return x5.mul(0.2) + x

class StyledRRDB(nn.Module):
    def __init__(self, nc, nrdb=3, kernel_size=3, gc=32, stride=1, dilation=1, 
                bias=True, act_type='leakyrelu', with_style=True, style_dim=512, with_noise=False):
        
        super(StyledRRDB, self).__init__()
        self.RDBs = nn.ModuleList([StyledResidualDenseBlock(nc, kernel_size=kernel_size, gc=gc, stride=stride, 
                                    dilation=dilation, bias=bias, act_type=act_type, with_style=with_style, 
                                    style_dim=style_dim, with_noise=with_noise) for _ in range(nrdb)])
        
    def forward(self, x, style):
        out = x
        for block in self.RDBs:
            out = block(out, style)
        return out.mul(0.2) + x

# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class UnetBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, in_nc, mid_nc, out_nc, bias=True, submodule=None, outermost=False, 
                    innermost=False, down_kernel=3, up_kernel=3):

        super(UnetBlock, self).__init__()
        self.outermost = outermost

        if in_nc is None:
            in_nc = out_nc

        downconv = nn.Conv2d(in_nc, mid_nc, kernel_size=down_kernel, stride=2, padding=1, bias=bias)
        downconv1 = nn.Conv2d(mid_nc, mid_nc, kernel_size=3, stride=1, padding=1, bias=bias)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.LeakyReLU(0.2, True)

        if outermost:
            upconv = B.upconvBlock(mid_nc * 2, mid_nc, kernel_size=3, stride=1, act_type='leakyrelu')
            upconv1 = nn.Conv2d(mid_nc, out_nc, kernel_size=up_kernel, stride=1, padding=1, bias=bias)
            down = [downconv, downrelu, downconv1]
            up = [uprelu, upconv, upconv1]
            self.model = nn.ModuleList(down + [submodule] + up)

        elif innermost:
            upconv = B.upconvBlock(mid_nc, out_nc, kernel_size=3, stride=1, act_type='leakyrelu')
            upconv1 = nn.Conv2d(mid_nc, out_nc, kernel_size=up_kernel, stride=1, padding=1, bias=bias)
            down = [downrelu, downconv, downrelu, downconv1]
            up = [uprelu, upconv, upconv1]
            self.model = nn.ModuleList(down + up)

        else:
            upconv = B.upconvBlock(mid_nc * 2, mid_nc, kernel_size=3, stride=1, act_type='leakyrelu')
            upconv1 = nn.Conv2d(mid_nc, out_nc, kernel_size=up_kernel, stride=1, padding=1, bias=bias)
            down = [downrelu, downconv, downrelu, downconv1]
            up = [uprelu, upconv, upconv1]

            self.model = nn.ModuleList(down + [submodule] + up)

    def forward(self, x, style=None):
        res = x
        for _, block in enumerate(self.model):
            res = block(res)

        if self.outermost:
            return res
        
        # print(x.shape, res.shape)
        return torch.cat([x, res], 1)

class UnetStyleBlock(nn.Module):

    def __init__(self, in_nc, mid_nc, out_nc, bias=True, submodule=None, outermost=False, 
                    innermost=False, with_style=True, style_dim=512, with_noise=False):

        super(UnetStyleBlock, self).__init__()
        self.outermost = outermost

        if in_nc is None:
            in_nc = out_nc

        relu = nn.LeakyReLU(0.2, True)
        if with_style:
            downconv = StyledconvBlock(in_nc, mid_nc, kernel_size=3, stride=2, with_noise=with_noise, act_type=relu)
            styleconv1 = StyledconvBlock(mid_nc, mid_nc, kernel_size=3, stride=1, with_noise=with_noise, act_type=None)
        else:
            downconv = B.convBlock(in_nc, mid_nc, kernel_size=3, stride=2, act_type='leakyrelu', inplace=False)
            styleconv1 = B.convBlock(mid_nc, mid_nc, kernel_size=3, stride=1, act_type=None)

        if outermost:
            if with_style:
                downconv = StyledconvBlock(in_nc, mid_nc, kernel_size=3, stride=2, with_noise=False, act_type=relu)
                upconv1 = StyledconvBlock(mid_nc * 2, mid_nc, kernel_size=3, stride=1, with_noise=with_noise, act_type=relu)
            else:
                downconv = B.convBlock(in_nc, mid_nc, kernel_size=3, stride=2, act_type='leakyrelu', inplace=False)
                upconv1 = B.convBlock(mid_nc * 2, mid_nc, kernel_size=3, stride=1, act_type='leakyrelu', inplace=False)

            upsample = B.upconvBlock(mid_nc * 2, mid_nc * 2)
            upout = B.convBlock(mid_nc, out_nc, kernel_size=3, stride=1, act_type=None)

            down = [downconv, styleconv1]
            up = [relu, upsample, upconv1, upout]
            self.model = nn.ModuleList(down + [submodule] + up)

        elif innermost:
            upsample = B.upconvBlock(mid_nc, mid_nc)

            if with_style:
                upout = StyledconvBlock(mid_nc, out_nc, kernel_size=3, stride=1, with_noise=with_noise, act_type=relu)
                upconv = StyledconvBlock(mid_nc, mid_nc, kernel_size=3, stride=1, with_noise=with_noise, act_type=relu)
            else:
                upconv = B.convBlock(mid_nc, mid_nc, kernel_size=3, stride=1, act_type='leakyrelu', inplace=False)
                upout = B.convBlock(mid_nc, out_nc, kernel_size=3, stride=1, act_type='leakyrelu', inplace=False)

            down = [relu, downconv, styleconv1]
            up = [relu, upsample, upconv, upout]
            self.model = nn.ModuleList(down + up)

        else:
            upsample = B.upconvBlock(mid_nc * 2, mid_nc * 2)

            if with_style:
                upconv1 = StyledconvBlock(mid_nc * 2, mid_nc, kernel_size=3, stride=1, with_noise=with_noise, act_type=relu)
                upout = StyledconvBlock(mid_nc, out_nc, kernel_size=3, stride=1, with_noise=with_noise, act_type=relu)
            else:
                upconv1 = B.convBlock(mid_nc * 2, mid_nc, kernel_size=3, stride=1, act_type='leakyrelu', inplace=False)
                upout = B.convBlock(mid_nc, out_nc, kernel_size=3, stride=1, act_type='leakyrelu', inplace=False)

            down = [relu, downconv, styleconv1]
            up = [relu, upsample, upconv1, upout]

            self.model = nn.ModuleList(down + [submodule] + up)

    def forward(self, x, style=None):
        res = x
        for _, block in enumerate(self.model):
            if isinstance(block, StyledconvBlock):
                res = block(res, style)
            else:
                res = block(res)

        if self.outermost:
            return res
        
        return torch.cat([x, res], 1)

class UnetRDBBlock(nn.Module):

    def __init__(self, in_nc, nf=64, act_type='leakyrelu', down_kernel=3, up_kernel=3, sub=None):
        
        super(UnetRDBBlock, self).__init__()
        self.downconv = nn.Conv2d(in_nc, nf, kernel_size=down_kernel, stride=2, padding=1)
        self.RDB1 = B.ResidualDenseBlock_5C(nf, gc=nf, act_type=act_type)
        self.sub = sub
        if sub is None:
            self.RDB2 = B.ResidualDenseBlock_5C(nf, gc=nf, act_type=act_type)
            self.upsample = B.upconvBlock(nf, nf, act_type=act_type)
        else:
            self.RDB2 = B.ResidualDenseBlock_5C(nf*2, gc=nf, act_type=act_type)
            self.upsample = B.upconvBlock(nf*2, nf, act_type=act_type)
        self.upconv = nn.Conv2d(in_nc, nf, kernel_size=up_kernel, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, True)
    
    def forward(self, x):
        out = self.lrelu(self.downconv(x))
        if not (self.sub is None):
            temp = self.sub(out)
            out = self.lrelu(self.RDB1(out))
            out = torch.cat((out, temp), 1)
        else:
            out = self.lrelu(self.RDB1(out))
        out = self.lrelu(self.RDB2(out))
        out = self.upconv(self.upsample(out))
        return out.mul(0.2) + x

class UnetRRDBBlock(nn.Module):

    def __init__(self, in_nc, nf=64, act_type='leakyrelu', down_kernel=3, up_kernel=3, sub=None):
        
        super(UnetRRDBBlock, self).__init__()
        self.downconv = nn.Conv2d(in_nc, nf, kernel_size=down_kernel, stride=2, padding=1)
        self.RRDB1 = B.RRDB(nf, gc=nf, act_type=act_type)
        self.sub = sub
        if sub is None:
            self.RRDB2 = B.RRDB(nf, gc=nf, act_type=act_type)
            self.upsample = B.upconvBlock(nf, nf, act_type=act_type)
        else:
            self.RRDB2 = B.RRDB(nf*2, gc=nf, act_type=act_type)
            self.upsample = B.upconvBlock(nf*2, nf, act_type=act_type)
        self.upconv = nn.Conv2d(in_nc, nf, kernel_size=up_kernel, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, True)
    
    def forward(self, x):
        out = self.lrelu(self.downconv(x))
        if not (self.sub is None):
            temp = self.sub(out)
            out = self.lrelu(self.RRDB1(out))
            out = torch.cat((out, temp), 1)
        else:
            out = self.lrelu(self.RRDB1(out))
        out = self.lrelu(self.RRDB2(out))
        out = self.upconv(self.upsample(out))
        return out.mul(0.2) + x

class UnetRDBBlock2(nn.Module):

    def __init__(self, in_nc, nf=64, act_type='leakyrelu', down_kernel=3, up_kernel=3, sub=None):
        
        super(UnetRDBBlock2, self).__init__()
        self.downconv = nn.Conv2d(in_nc, nf, kernel_size=down_kernel, stride=2, padding=1)
        self.RDB1 = B.ResidualDenseBlock_5C(nf, gc=nf, act_type=act_type)
        self.sub = sub
        if sub is None:
            self.RDB2 = B.ResidualDenseBlock_5C(nf, gc=nf, act_type=act_type)
            self.upsample = B.upconvBlock(nf, nf, act_type=act_type)
        else:
            self.RDB2 = B.ResidualDenseBlock_5C(nf, gc=nf, act_type=act_type)
            self.upsample = B.upconvBlock(nf, nf, act_type=act_type)
        self.upconv = nn.Conv2d(in_nc, nf, kernel_size=up_kernel, stride=1, padding=1)
    
    def forward(self, x):
        out = self.downconv(x)
        temp1 = out
        temp2 = 0
        if not (self.sub is None):
            temp2 = self.sub(out)
        out = self.RDB1(out)
        out = temp2 + out
        out = self.RDB2(out)
        out = temp1 + out
        out = self.upconv(self.upsample(out))
        return out.mul(0.2) + x

class UnetRDBBlock3(nn.Module):

    def __init__(self, in_nc, nf=64, act_type='leakyrelu', down_kernel=3, up_kernel=3, sub=None):
        
        super(UnetRDBBlock3, self).__init__()
        self.downconv = nn.Conv2d(in_nc, nf, kernel_size=down_kernel, stride=2, padding=1)
        self.RDB1 = B.ResidualDenseBlock_5C(nf, gc=nf, act_type=act_type)
        self.sub = sub
        self.RDB2 = B.ResidualDenseBlock_5C(nf, gc=nf, act_type=act_type)
        self.upsample = B.upconvBlock(nf, nf, act_type=act_type)
        self.RDB3 = B.ResidualDenseBlock_5C(nf, gc=nf, act_type=act_type)
        self.upconv = nn.Conv2d(in_nc, nf, kernel_size=up_kernel, stride=1, padding=1)
    
    def forward(self, x):
        out = self.downconv(x)
        temp1 = out
        out = self.RDB1(out)
        out = temp1 + self.RDB2(out)

        temp2 = 0
        if not (self.sub is None):
            temp2 = self.sub(temp1)

        out = temp2 + out
        out = self.upsample(out)
        out = self.RDB3(out)
        out = self.upconv(out)
        return out.mul(0.2) + x

class UnetRDBBlock4(nn.Module):

    def __init__(self, in_nc, nf=64, act_type='leakyrelu', down_kernel=3, up_kernel=3, sub=None):
        
        super(UnetRDBBlock4, self).__init__()
        self.downconv = nn.Conv2d(in_nc, nf, kernel_size=down_kernel, stride=2, padding=1)
        self.RDB1 = B.ResidualDenseBlock_5C(nf, gc=nf, act_type=act_type)
        self.sub = sub
        self.RDB2 = B.ResidualDenseBlock_5C(nf, gc=nf, act_type=act_type)
        if not (self.sub is None):
            self.catconv = B.convBlock(nf*3, nf, kernel_size=3, act_type=act_type)
        else:
            self.catconv = B.convBlock(nf*2, nf, kernel_size=3, act_type=act_type)
        self.upsample = B.upconvBlock(nf, nf, act_type=act_type)
        self.RDB3 = B.ResidualDenseBlock_5C(nf, gc=nf, act_type=act_type)
        self.upconv = nn.Conv2d(in_nc, nf, kernel_size=up_kernel, stride=1, padding=1)
    
    def forward(self, x):
        out = self.downconv(x)
        temp1 = out
        out = self.RDB1(out)
        out = self.RDB2(out)

        temp2 = 0
        if not (self.sub is None):
            temp2 = self.sub(temp1)
            out = torch.cat((out, temp2, temp1), 1)
        else:
            out = torch.cat((out, temp1), 1)
        out = self.catconv(out)

        out = self.upsample(out)
        out = self.RDB3(out)
        out = self.upconv(out)
        return out.mul(0.2) + x

class UnetRRDBBlock2(nn.Module):

    def __init__(self, in_nc, nf=64, act_type='leakyrelu', down_kernel=3, up_kernel=3, sub=None):
        
        super(UnetRRDBBlock2, self).__init__()
        self.downconv = nn.Conv2d(in_nc, nf, kernel_size=down_kernel, stride=2, padding=1)
        self.RDB1 = B.RRDB(nf, gc=nf, act_type=act_type)
        self.sub = sub
        self.RDB2 = B.RRDB(nf, gc=nf, act_type=act_type)
        self.upsample = B.upconvBlock(nf, nf, act_type=act_type)
        self.RDB3 = B.RRDB(nf, gc=nf, act_type=act_type)
        self.upconv = nn.Conv2d(in_nc, nf, kernel_size=up_kernel, stride=1, padding=1)
    
    def forward(self, x):
        out = self.downconv(x)
        temp1 = out
        out = self.RDB1(out)
        out = temp1 + self.RDB2(out)

        temp2 = 0
        if not (self.sub is None):
            temp2 = self.sub(temp1)

        out = temp2 + out
        out = self.upsample(out)
        out = self.RDB3(out)
        out = self.upconv(out)
        return out.mul(0.2) + x

class RRDBUnetSkipBlock(nn.Module):

    def __init__(self, nb=4, nf=64, norm_type=None, act_type='leakyrelu'):
        
        super(RRDBUnetSkipBlock, self).__init__()
        self.blocks = nn.ModuleList([B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero',
                                    norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)])
        unetRDBBlock = UnetRDBBlock2(nf, nf=nf, act_type=act_type, up_kernel=(3, 4), sub=None)
        unetRDBBlock = UnetRDBBlock2(nf, nf=nf, act_type=act_type, down_kernel=(2, 3), up_kernel=(4, 3), sub=unetRDBBlock)
        unetRDBBlock = UnetRDBBlock2(nf, nf=nf, act_type=act_type, sub=unetRDBBlock)
        self.unet = UnetRDBBlock2(nf, nf=nf, act_type=act_type, sub=unetRDBBlock)
    
    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out + self.unet(x)

class RRDBUnetSkipBlock2(nn.Module):

    def __init__(self, nb=4, nf=64, norm_type=None, act_type='leakyrelu'):
        
        super(RRDBUnetSkipBlock2, self).__init__()
        self.blocks = nn.ModuleList([B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero',
                                    norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)])
        unetRDBBlock = UnetRDBBlock3(nf, nf=nf, act_type=act_type, sub=None)
        self.unet = UnetRDBBlock3(nf, nf=nf, act_type=act_type, sub=unetRDBBlock)
    
    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out + self.unet(x)

class RRDBUnetSkipBlock3(nn.Module):

    def __init__(self, nb=4, nf=64, norm_type=None, act_type='leakyrelu'):
        
        super(RRDBUnetSkipBlock3, self).__init__()
        self.blocks = nn.ModuleList([B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero',
                                    norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)])
        unetRDBBlock = UnetRDBBlock4(nf, nf=nf, act_type=act_type, sub=None)
        self.unet = UnetRDBBlock4(nf, nf=nf, act_type=act_type, sub=unetRDBBlock)
    
    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out + self.unet(x)