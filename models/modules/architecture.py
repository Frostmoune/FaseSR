import math
import torch
import torch.nn as nn
from . import block as B
import torchvision
from torch.nn import Parameter, init
from . import spectral_norm as SN
from . import resnet_block as RB
from . import style_block as SB


# Generator

class SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
            mode='CNA', res_scale=1, upsample_mode='upconv'):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x

class RRDBNet_FSR(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
            mode='CNA', res_scale=1, upsample_mode='upconv', with_prior=False):
        super(RRDBNet_FSR, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = [B.convBlock(in_nc, nf, kernel_size=3, norm_type=None, act_type=None, mode=mode)]
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(4)]
        LR_conv = [B.convBlock(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)]
        self.coarse = nn.ModuleList(fea_conv + rb_blocks + LR_conv)
        
        if with_prior:
            rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(4)]
            self.prior = nn.ModuleList(LR_conv + rb_blocks + LR_conv)
        else:
            self.prior = None
        
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb - 8)]
        self.encoder = nn.ModuleList(LR_conv + rb_blocks + LR_conv)

        rb_blocks = [B.RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(4)]
        LR_conv = [B.convBlock(nf * 2, nf * 2, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)]
        self.decoder = nn.ModuleList(LR_conv + rb_blocks + LR_conv)

        if upsample_mode == 'upconv':
            upsample_block = B.upconvBlock
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffleBlock
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            upsampler = [upsample_block(nf * 2, nf * 2, 3, act_type=act_type)]
        else:
            upsampler = [upsample_block(nf * 2, nf * 2, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = [B.convBlock(nf * 2, nf * 2, kernel_size=3, norm_type=None, act_type=act_type)]
        HR_conv1 = [B.convBlock(nf * 2, out_nc, kernel_size=3, norm_type=None, act_type=None)]

        self.upsample = nn.ModuleList(upsampler + HR_conv0 + HR_conv1)
    
    def forward(self, x):
        for block in self.coarse:
            x = block(x)
        temp = x

        if not (self.prior is None):
            for block in self.prior:
                temp = block(temp)
        
        for block in self.encoder:
            x = block(x)
        
        x = torch.cat((x, temp), 1)
        for block in self.decoder:
            x = block(x)
        
        for block in self.upsample:
            x = block(x)
        
        return x

# Add RRDB_AdaIn
class RRDBNet_AdaIn(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
            mode='CNA', res_scale=1, upsample_mode='upconv'):
        super(RRDBNet_AdaIn, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        self.fea_conv = B.convBlock(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        self.rb_blocks = nn.ModuleList([SB.RRDB_AdaIn(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)])
        self.LR_conv = B.convBlock(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconvBlock
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffleBlock
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        if upscale == 3:
            self.upsampler = nn.ModuleList([upsample_block(nf, nf, 3, act_type=act_type)])
        else:
            self.upsampler = nn.ModuleList([upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)])
        self.HR_conv0 = B.convBlock(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv1 = B.convBlock(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

    def forward(self, x, style):
        x = self.fea_conv(x, style)

        temp = x
        for block in self.rb_blocks:
            x = block(x, style)
        x = temp + self.LR_conv(x, style)

        for block in self.upsampler:
            x = block(x, style)

        x = self.HR_conv0(x, style)
        x = self.HR_conv1(x, style)

        return x


class RRDBUNet(nn.Module):

    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', 
                    mode='CNA', upsample_mode='upconv', with_style=True, with_unet=True):
        super(RRDBUNet, self).__init__()

        if upsample_mode == 'upconv':
            upsample_block = B.upconvBlock
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffleBlock
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        
        nb_2x = nb // 3
        nb_4x = nb - nb_2x

        self.fea_conv = B.convBlock(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        self.block_2x = nn.ModuleList([B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb_2x)])
        self.conv_2x = B.convBlock(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type, mode=mode)
        self.upsample_2x = upsample_block(nf, nf, act_type=act_type)
        
        self.block_4x = nn.ModuleList([B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb_4x)])
        self.conv_4x = B.convBlock(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.upsample_4x = upsample_block(nf, nf, act_type=act_type)

        self.toHR = nn.ModuleList([
                    B.convBlock(nf, nf, kernel_size=3, norm_type=None, act_type=act_type),
                    B.convBlock(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)])
        self.unet1, self.unet2 = None, None

        if with_unet:
            unet_block = SB.UnetBlock(in_nc=None, mid_nc=nf * 4, out_nc=nf * 4, submodule=None, innermost=True, up_kernel=(3, 4))
            unet_block = SB.UnetBlock(in_nc=None, mid_nc=nf * 4, out_nc=nf * 2, submodule=unet_block, down_kernel=(2, 3), up_kernel=(4, 3))
            unet_block = SB.UnetBlock(in_nc=None, mid_nc=nf * 2, out_nc=nf, submodule=unet_block)
            self.unet1 = SB.UnetBlock(in_nc=nf, mid_nc=nf, out_nc=nf, submodule=unet_block, outermost=True)
            self.unet_lrelu = nn.LeakyReLU(0.2, True)

            unet_block = SB.UnetBlock(in_nc=None, mid_nc=nf * 8, out_nc=nf * 8, submodule=None, innermost=True, up_kernel=(3, 4))
            unet_block = SB.UnetBlock(in_nc=None, mid_nc=nf * 8, out_nc=nf * 8, submodule=unet_block, down_kernel=(2, 3), up_kernel=(4, 3))
            unet_block = SB.UnetBlock(in_nc=None, mid_nc=nf * 8, out_nc=nf * 4, submodule=unet_block)
            unet_block = SB.UnetBlock(in_nc=None, mid_nc=nf * 4, out_nc=nf * 2, submodule=unet_block)
            self.unet2 = SB.UnetBlock(in_nc=nf * 2, mid_nc=nf * 2, out_nc=nf * 2, submodule=unet_block, outermost=True)

            self.upsample_2x = upsample_block(nf * 2, nf * 2, act_type=act_type)
            self.block_4x = nn.ModuleList([B.RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb_4x)])
            self.conv_4x = B.convBlock(nf * 2, nf * 2, kernel_size=3, norm_type=None, act_type=act_type)
            self.upsample_4x = upsample_block(nf * 4, nf * 4, act_type=act_type)

            self.toHR = nn.ModuleList([B.convBlock(nf * 4, nf * 2, kernel_size=3, norm_type=None, act_type=act_type),
                        B.convBlock(nf * 2, nf, kernel_size=3, norm_type=None, act_type=act_type),
                        B.convBlock(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)])

    def forward(self, x):
        x = self.fea_conv(x)
        temp = x
        for block in self.block_2x:
            x = block(x)
        x = temp + self.conv_2x(x)
        if not (self.unet1 is None):
            x = torch.cat((x, self.unet_lrelu(self.unet1(temp))), 1)
        x = self.upsample_2x(x)

        temp = x
        for block in self.block_4x:
            x = block(x)
        x = temp + self.conv_4x(x)
        if not (self.unet2 is None):
            x = torch.cat((x, self.unet_lrelu(self.unet2(temp))), 1)
        x = self.upsample_4x(x)

        for block in self.toHR:
            x = block(x)

        return x

class RRDBUNet2(nn.Module):

    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', 
                    mode='CNA', upsample_mode='upconv'):
        super(RRDBUNet2, self).__init__()

        if upsample_mode == 'upconv':
            upsample_block = B.upconvBlock
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffleBlock
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)

        self.fea_conv = B.convBlock(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        self.blocks = nn.ModuleList([SB.RRDB_Unet(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb)])
        self.conv_LR = B.convBlock(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.upsample_2x = upsample_block(nf, nf, act_type=act_type)
        self.upsample_4x = upsample_block(nf, nf, act_type=act_type)

        self.conv_4x = B.convBlock(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.to_rgb = B.convBlock(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

    def forward(self, x):
        x = self.fea_conv(x)
        temp = x
        for block in self.blocks:
            x = block(x)
        x = temp + self.conv_LR(x)
        x = self.upsample_2x(x)
        x = self.upsample_4x(x)
        x = self.conv_4x(x)
        x = self.to_rgb(x)

        return x

class RRDBUNet3(nn.Module):

    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', 
                    mode='CNA', upsample_mode='upconv', with_style=True, with_unet=True):
        super(RRDBUNet3, self).__init__()

        if upsample_mode == 'upconv':
            upsample_block = B.upconvBlock
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffleBlock
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        
        nb_2x = nb // 3
        nb_4x = nb - nb_2x

        self.fea_conv = B.convBlock(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        self.block_2x = nn.ModuleList([B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb_2x)])
        self.conv_2x = B.convBlock(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type, mode=mode)
        self.upsample_2x = upsample_block(nf * 2, nf * 2, act_type=act_type)
        
        self.block_4x = nn.ModuleList([B.RRDB(nf * 2, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
                                norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb_4x)])
        self.conv_4x = B.convBlock(nf * 2, nf * 2, kernel_size=3, norm_type=None, act_type=act_type)
        self.upsample_4x = upsample_block(nf * 4, nf * 4, act_type=act_type)

        self.toHR = nn.ModuleList([
                    B.convBlock(nf * 4, nf, kernel_size=3, norm_type=None, act_type=act_type),
                    B.convBlock(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)])
                    
        unetRDBBlock = SB.UnetRDBBlock(nf, nf=nf, act_type=act_type, up_kernel=(3, 4), sub=None)
        unetRDBBlock = SB.UnetRDBBlock(nf, nf=nf, act_type=act_type, down_kernel=(2, 3), up_kernel=(4, 3), sub=unetRDBBlock)
        unetRDBBlock = SB.UnetRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRDBBlock)
        self.unet1 =  SB.UnetRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRDBBlock)

        unetRDBBlock = SB.UnetRDBBlock(nf*2, nf=nf*2, act_type=act_type, up_kernel=(3, 4), sub=None)
        unetRDBBlock = SB.UnetRDBBlock(nf*2, nf=nf*2, act_type=act_type, down_kernel=(2, 3), up_kernel=(4, 3), sub=unetRDBBlock)
        unetRDBBlock = SB.UnetRDBBlock(nf*2, nf=nf*2, act_type=act_type, sub=unetRDBBlock)
        unetRDBBlock = SB.UnetRDBBlock(nf*2, nf=nf*2, act_type=act_type, sub=unetRDBBlock)
        self.unet2 =  SB.UnetRDBBlock(nf*2, nf=nf*2, act_type=act_type, sub=unetRDBBlock)


    def forward(self, x):
        x = self.fea_conv(x)
        temp = x
        for block in self.block_2x:
            x = block(x)
        x = temp + self.conv_2x(x)
        if not (self.unet1 is None):
            x = torch.cat((x, self.unet1(temp)), 1)
        x = self.upsample_2x(x)

        temp = x
        for block in self.block_4x:
            x = block(x)
        x = temp + self.conv_4x(x)
        if not (self.unet2 is None):
            x = torch.cat((x, self.unet2(temp)), 1)
        x = self.upsample_4x(x)

        for block in self.toHR:
            x = block(x)

        return x


# Add StyleRRDB
class StyleRRDBNet(nn.Module):

    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', 
                    mode='CNA', upsample_mode='upconv', with_style=True, with_noise=False):
        super(StyleRRDBNet, self).__init__()

        if upsample_mode == 'upconv':
            upsample_block = B.upconvBlock
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffleBlock
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        
        nb_2x = nb // 3
        nb_4x = nb - nb_2x
        self.fea_conv = B.convBlock(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        self.block_2x = nn.ModuleList([SB.StyledRRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, 
                            act_type=act_type, with_style=with_style, with_noise=with_noise) for _ in range(nb_2x)])
        self.conv_2x = B.convBlock(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
        self.upsample_2x = upsample_block(nf, nf, act_type=act_type)
        
        self.block_4x = nn.ModuleList([SB.StyledRRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, 
                            act_type=act_type, with_style=with_style, with_noise=with_noise) for _ in range(nb_4x)])
        self.upsample_4x = upsample_block(nf, nf, act_type=act_type)

        self.conv_4x = B.convBlock(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.conv_HR = B.convBlock(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

    def forward(self, x, style):
        x = self.fea_conv(x)
        temp = x
        for block in self.block_2x:
            x = block(x, style)
        x = temp + self.conv_2x(x)

        x = self.upsample_2x(x)
        temp = x
        for block in self.block_4x:
            x = block(x, style)
        x = temp + self.conv_4x(x)

        x = self.upsample_4x(x)
        x = self.conv_HR(x)

        return x

# Add StyleRRDB
class StyleResNet(nn.Module):

    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type=None, act_type='leakyrelu', 
                    mode='CNA', upsample_mode='upconv', with_style=True, with_noise=False):
        super(StyleResNet, self).__init__()

        if upsample_mode == 'upconv':
            upsample_block = B.upconvBlock
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffleBlock
        else:
            raise NotImplementedError('upsample mode [%s] is not found' % upsample_mode)
        
        nb_2x = nb // 3
        nb_4x = nb - nb_2x
        self.fea_conv_2x = B.convBlock(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        self.block_2x = nn.ModuleList([SB.StyledResNetBlock(nf, nf, nf, kernel_size=3, stride=1, bias=True, 
                            act_type=act_type, with_style=with_style, with_noise=with_noise) for _ in range(nb_2x)])
        self.conv_2x = B.convBlock(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type, mode=mode)
        self.upsample_2x = upsample_block(nf, nf, act_type=act_type)
        self.torgb_2x = B.convBlock(nf, out_nc, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
        
        self.fea_conv_4x = B.convBlock(out_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        self.block_4x = nn.ModuleList([SB.StyledResNetBlock(nf, nf, nf, kernel_size=3, stride=1, bias=True, 
                            act_type=act_type, with_style=with_style, with_noise=with_noise) for _ in range(nb_4x)])
        self.conv_4x = B.convBlock(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.upsample_4x = upsample_block(nf, nf, act_type=act_type)

        self.conv_HR = B.convBlock(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.torgb_4x = B.convBlock(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

    def forward(self, x, style):
        x = self.fea_conv_2x(x)
        temp1 = x
        for block in self.block_2x:
            x = block(x, style)
        x = temp1 + self.conv_2x(x)
        x = self.upsample_2x(x)
        x = self.torgb_2x(x)

        x = self.fea_conv_4x(x)
        temp2 = x
        for block in self.block_4x:
            x = block(x, style)
        x = temp2 + self.conv_4x(x)
        x = self.upsample_4x(x)

        x = self.conv_HR(x)
        x = self.torgb_4x(x)

        return x

class Unet(nn.Module):

    def __init__(self, in_nc, out_nc, nf=64):
        super(Unet, self).__init__()
        # construct unet structure
        unet_block = SB.UnetBlock(in_nc=None, mid_nc=nf * 8, out_nc=nf * 8, submodule=None, innermost=True, up_kernel=(3, 4))  # add the innermost layer
        unet_block = SB.UnetBlock(in_nc=None, mid_nc=nf * 8, out_nc=nf * 8, submodule=unet_block, down_kernel=(2, 3), up_kernel=(4, 3))
        unet_block = SB.UnetBlock(in_nc=None, mid_nc=nf * 8, out_nc=nf * 4, submodule=unet_block)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = SB.UnetBlock(in_nc=None, mid_nc=nf * 4, out_nc=nf * 2, submodule=unet_block)
        unet_block = SB.UnetBlock(in_nc=None, mid_nc=nf * 2, out_nc=nf, submodule=unet_block)
        self.model = SB.UnetBlock(in_nc=in_nc, mid_nc=nf, out_nc=out_nc, submodule=unet_block, outermost=True)  # add the outermost layer

    def forward(self, x, style=None):
        return self.model(x, style)

class UnetStyle(nn.Module):
    def __init__(self, in_nc, out_nc, nf=64, 
                with_style=True, style_dim=512, with_noise=False):
        super(UnetStyle, self).__init__()
        # construct unet structure
        unet_block = SB.UnetStyleBlock(in_nc=None, mid_nc=nf * 4, out_nc=nf * 4, submodule=None, innermost=True, 
                                        with_style=with_style, style_dim=style_dim, with_noise=with_noise)  # add the innermost layer
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = SB.UnetStyleBlock(in_nc=None, mid_nc=nf * 4, out_nc=nf * 2, submodule=unet_block, 
                                        with_style=with_style, style_dim=style_dim, with_noise=with_noise)
        unet_block = SB.UnetStyleBlock(in_nc=None, mid_nc=nf * 2, out_nc=nf, submodule=unet_block, 
                                        with_style=with_style, style_dim=style_dim, with_noise=with_noise)
        self.model = SB.UnetStyleBlock(in_nc=in_nc, mid_nc=nf, out_nc=out_nc, submodule=unet_block, outermost=True,
                                        with_style=with_style, style_dim=style_dim, with_noise=with_noise)  # add the outermost layer
    
    def forward(self, x, style=None):
        return self.model(x, style)

class UnetRDBNet(nn.Module):

    def __init__(self, in_nc, out_nc, nf=64, act_type='leakyrelu'):
        super(UnetRDBNet, self).__init__()
        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, bias=True, padding=1)
        unetRDBBlock = SB.UnetRDBBlock(nf, nf=nf, act_type=act_type, up_kernel=(3, 4), sub=None)
        unetRDBBlock = SB.UnetRDBBlock(nf, nf=nf, act_type=act_type, down_kernel=(2, 3), up_kernel=(4, 3), sub=unetRDBBlock)
        unetRDBBlock = SB.UnetRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRDBBlock)
        unetRDBBlock = SB.UnetRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRDBBlock)
        unetRDBBlock = SB.UnetRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRDBBlock)
        self.unet = SB.UnetRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRDBBlock)
        self.to_rgb = nn.Conv2d(nf, out_nc, kernel_size=3, stride=1, bias=True, padding=1)
    
    def forward(self, x, style=None):
        x = self.fea_conv(x)
        x = self.unet(x)
        return self.to_rgb(x)

class UnetRRDBNet(nn.Module):

    def __init__(self, in_nc, out_nc, nf=64, act_type='leakyrelu'):
        super(UnetRRDBNet, self).__init__()
        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, bias=True, padding=1)
        unetRRDBBlock = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, up_kernel=(3, 4), sub=None)
        unetRRDBBlock = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, down_kernel=(2, 3), up_kernel=(4, 3), sub=unetRRDBBlock)
        unetRRDBBlock = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRRDBBlock)
        unetRRDBBlock = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRRDBBlock)
        unetRRDBBlock = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRRDBBlock)
        self.unet = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRRDBBlock)
        self.to_rgb = nn.Conv2d(nf, out_nc, kernel_size=3, stride=1, bias=True, padding=1)
    
    def forward(self, x, style=None):
        x = self.fea_conv(x)
        x = self.unet(x)
        return self.to_rgb(x)

class UnetRRDBNet2(nn.Module):

    def __init__(self, in_nc, out_nc, nf=64, act_type='leakyrelu'):
        super(UnetRRDBNet2, self).__init__()
        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, bias=True, padding=1)
        unetRRDBBlock = SB.UnetRRDBBlock2(nf, nf=nf, act_type=act_type, sub=None)
        unetRRDBBlock = SB.UnetRRDBBlock2(nf, nf=nf, act_type=act_type, sub=unetRRDBBlock)
        unetRRDBBlock = SB.UnetRRDBBlock2(nf, nf=nf, act_type=act_type, sub=unetRRDBBlock)
        self.unet = SB.UnetRRDBBlock2(nf, nf=nf, act_type=act_type, sub=unetRRDBBlock)
        self.to_rgb = nn.Conv2d(nf, out_nc, kernel_size=3, stride=1, bias=True, padding=1)
    
    def forward(self, x, style=None):
        x = self.fea_conv(x)
        x = self.unet(x)
        return self.to_rgb(x)

class RRDBUNet4(nn.Module):

    def __init__(self, in_nc, out_nc, nf=64, act_type='leakyrelu'):
        super(RRDBUNet4, self).__init__()
        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, bias=True, padding=1)
        unetRRDBBlock = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, up_kernel=(3, 4), sub=None)
        unetRRDBBlock = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, down_kernel=(2, 3), up_kernel=(4, 3), sub=unetRRDBBlock)
        unetRRDBBlock = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRRDBBlock)
        self.unet1 = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRRDBBlock)
        self.upsample1 = B.upconvBlock(nf, nf, act_type=act_type)
        unetRRDBBlock = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, up_kernel=(3, 4), sub=None)
        unetRRDBBlock = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, down_kernel=(2, 3), up_kernel=(4, 3), sub=unetRRDBBlock)
        unetRRDBBlock = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRRDBBlock)
        unetRRDBBlock = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRRDBBlock)
        self.unet2 = SB.UnetRRDBBlock(nf, nf=nf, act_type=act_type, sub=unetRRDBBlock)
        self.upsample2 = B.upconvBlock(nf, nf, act_type=act_type)
        self.to_rgb = nn.Conv2d(nf, out_nc, kernel_size=3, stride=1, bias=True, padding=1)
    
    def forward(self, x, style=None):
        x = self.fea_conv(x)
        x = self.unet1(x)
        x = self.upsample1(x)
        x = self.unet2(x)
        x = self.upsample2(x)
        return self.to_rgb(x)

class RRDBUNet5(nn.Module):

    def __init__(self, in_nc, out_nc, na, nb=4, nf=64, norm_type=None, act_type='leakyrelu'):
        super(RRDBUNet5, self).__init__()
        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, bias=True, padding=1)
        self.blocks = nn.ModuleList([SB.RRDBUnetSkipBlock(nb, nf, norm_type=norm_type, act_type=act_type) for _ in range(na)])
        self.upsample = nn.ModuleList([B.upconvBlock(nf, nf, act_type=act_type), B.upconvBlock(nf, nf, act_type=act_type)])
        self.to_rgb = nn.Conv2d(nf, out_nc, kernel_size=3, stride=1, bias=True, padding=1)
    
    def forward(self, x, style=None):
        out = self.fea_conv(x)
        temp = out
        for block in self.blocks:
            out = block(out)
        out = temp + out
        for block in self.upsample:
            out = block(out)
        return self.to_rgb(out)

class RRDBUNet6(nn.Module):

    def __init__(self, in_nc, out_nc, na, nb=4, nf=64, norm_type=None, act_type='leakyrelu'):
        super(RRDBUNet6, self).__init__()
        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, bias=True, padding=1)
        self.blocks = nn.ModuleList([SB.RRDBUnetSkipBlock2(nb, nf, norm_type=norm_type, act_type=act_type) for _ in range(na)])
        self.upsample = nn.ModuleList([B.upconvBlock(nf, nf, act_type=act_type), B.upconvBlock(nf, nf, act_type=act_type)])
        self.to_rgb = nn.Conv2d(nf, out_nc, kernel_size=3, stride=1, bias=True, padding=1)
    
    def forward(self, x, style=None):
        out = self.fea_conv(x)
        temp = out
        for block in self.blocks:
            out = block(out)
        out = temp + out
        for block in self.upsample:
            out = block(out)
        return self.to_rgb(out)

class RRDBUNet7(nn.Module):

    def __init__(self, in_nc, out_nc, na, nb=4, nf=64, norm_type=None, act_type='leakyrelu'):
        super(RRDBUNet7, self).__init__()
        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, bias=True, padding=1)
        self.blocks = nn.ModuleList([SB.RRDBUnetSkipBlock3(nb, nf, norm_type=norm_type, act_type=act_type) for _ in range(na)])
        self.upsample = nn.ModuleList([B.upconvBlock(nf, nf, act_type=act_type), B.upconvBlock(nf, nf, act_type=act_type)])
        self.to_rgb = nn.Conv2d(nf, out_nc, kernel_size=3, stride=1, bias=True, padding=1)
    
    def forward(self, x, style=None):
        out = self.fea_conv(x)
        temp = out
        for block in self.blocks:
            out = block(out)
        out = temp + out
        for block in self.upsample:
            out = block(out)
        return self.to_rgb(out)
        
# Discriminator

# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 64, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 32, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 16, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 8, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 4, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# VGG style Discriminator with input size 128*128, Spectral Normalization
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self, in_nc):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = SN.spectral_norm(nn.Conv2d(in_nc, 64, 3, 1, 1))
        self.conv1 = SN.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = SN.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = SN.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = SN.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = SN.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = SN.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = SN.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = SN.spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = SN.spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x

class Discriminator_VGG_96(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 6, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# VGG style Discriminator with input size 96*96, Spectral Normalization
class Discriminator_VGG_96_SN(nn.Module):
    def __init__(self, in_nc):
        super(Discriminator_VGG_96_SN, self).__init__()
        # features
        # hxw, c
        # 96, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = SN.spectral_norm(nn.Conv2d(in_nc, 64, 3, 1, 1))
        self.conv1 = SN.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 48, 64
        self.conv2 = SN.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = SN.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 24, 128
        self.conv4 = SN.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = SN.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 12, 256
        self.conv6 = SN.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 6, 512
        self.conv8 = SN.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 3, 512

        # classifier
        self.linear0 = SN.spectral_norm(nn.Linear(512 * 3 * 3, 100))
        self.linear1 = SN.spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x

class Discriminator_VGG_Feature(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_Feature, self).__init__()
        # features
        # hxw, c
        # 32 * 16, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 16 * 8, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 8 * 4, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 4 * 2, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 2 * 1, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 1, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# VGG style Discriminator with input size 32*16, Spectral Normalization
class Discriminator_VGG_Feature_SN(nn.Module):
    def __init__(self, in_nc):
        super(Discriminator_VGG_Feature_SN, self).__init__()
        # features
        # hxw, c
        # 32 * 16, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = SN.spectral_norm(nn.Conv2d(in_nc, 64, 3, 1, 1))
        self.conv1 = SN.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 16 * 8, 64
        self.conv2 = SN.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = SN.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 8 * 4, 128
        self.conv4 = SN.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = SN.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 4 * 2, 256
        self.conv6 = SN.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))

        # 2 * 1, 512
        # classifier
        self.linear0 = SN.spectral_norm(nn.Linear(512 * 2 * 1, 100))
        self.linear1 = SN.spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x

class Discriminator_VGG_192(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 96, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 6, 512
        conv10 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv11 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9, conv10, conv11)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Discriminator_Feature(nn.Module):

    def __init__(self, nb=8, feature_dim=512):
        super(Discriminator_Feature, self).__init__()

        self.net = []
        for _ in range(nb):
            self.net.append(nn.Linear(feature_dim, feature_dim))
            self.net.append(nn.LeakyReLU(0.2))
        
        self.net.append(nn.Linear(feature_dim, 100))
        self.net.append(nn.LeakyReLU(0.2))
        self.net.append(nn.Linear(100, 1))
        self.net = nn.ModuleList(self.net)
    
    def forward(self, feature):
        for block in self.net:
            feature = block(feature)
        return feature

class Discriminator_SNResNetProjection_96(nn.Module):

    def __init__(self, in_nc, num_features=64):
        super(Discriminator_SNResNetProjection_96, self).__init__()
        self.num_features = num_features
        self.activation = nn.LeakyReLU(0.2, inplace=False)

        self.block1 = RB.ResOptimizedBlock(in_nc, num_features)
        self.block2 = RB.ResBlock(num_features, num_features * 2,
                            activation=self.activation, downsample=True)
        self.block3 = RB.ResBlock(num_features * 2, num_features * 4,
                            activation=self.activation, downsample=True)
        self.block4 = RB.ResBlock(num_features * 4, num_features * 8,
                            activation=self.activation, downsample=True)
        self.block5 = RB.ResBlock(num_features * 8, num_features * 16,
                            activation=self.activation, downsample=True)
        self.l6 = SN.spectral_norm(nn.Linear(num_features * 16, 1))

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)
        return output

class Discriminator_SNResNetProjection_Feature(nn.Module):

    def __init__(self, in_nc, num_features=64):
        super(Discriminator_SNResNetProjection_Feature, self).__init__()
        self.num_features = num_features
        self.activation = nn.LeakyReLU(0.2, inplace=False)

        self.block1 = RB.ResOptimizedBlock(in_nc, num_features)
        self.block2 = RB.ResBlock(num_features, num_features * 2,
                            activation=self.activation, downsample=True)
        self.block3 = RB.ResBlock(num_features * 2, num_features * 4,
                            activation=self.activation, downsample=True)
        self.block4 = RB.ResBlock(num_features * 4, num_features * 8,
                            activation=self.activation, downsample=True)
        self.l5 = SN.spectral_norm(nn.Linear(num_features * 8, 1))

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)
        return output

class Discriminator_Sphere20a(nn.Module):
    def __init__(self,
                 model_path,
                 use_input_norm=True,
                 device=torch.device('cuda')):
        super(Discriminator_Sphere20a, self).__init__()
        model = Sphere20a()
        
        from collections import OrderedDict
        state_dict_remove_fc = OrderedDict()
        for x, y in torch.load(model_path).items():
            if x[:3] != 'fc6':
                state_dict_remove_fc[x] = y
        model.load_state_dict(state_dict_remove_fc)

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        self.features = model
        self.linear1 = nn.Linear(512, 100)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.linear2 = nn.Linear(100, 1)

        init.kaiming_normal_(self.linear2.weight.data, a=0, mode='fan_in')
        self.linear1.bias.data.zero_()
        init.kaiming_normal_(self.linear1.weight.data, a=0, mode='fan_in')
        self.linear2.bias.data.zero_()

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        out = self.features(x)
        out = self.linear1(out)
        out = self.lrelu(out)
        out = self.linear2(out)
        return out

# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cuda')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

# Add VGG16
class VGG16FeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=28,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cuda')):
        super(VGG16FeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg16_bn(pretrained=True)
        else:
            model = torchvision.models.vgg16(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

class MINCNet(nn.Module):
    def __init__(self):
        super(MINCNet, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv1_1(x))
        out = self.ReLU(self.conv1_2(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv2_1(out))
        out = self.ReLU(self.conv2_2(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv3_1(out))
        out = self.ReLU(self.conv3_2(out))
        out = self.ReLU(self.conv3_3(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv4_1(out))
        out = self.ReLU(self.conv4_2(out))
        out = self.ReLU(self.conv4_3(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv5_1(out))
        out = self.ReLU(self.conv5_2(out))
        out = self.conv5_3(out)
        return out

# Add VGG16-MINC
class VGG16MINCFeatureExtractor(nn.Module):
    def __init__(self,
                 model_path,
                 feature_layer=28,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cuda')):
        super(VGG16MINCFeatureExtractor, self).__init__()
        model = MINCNet()
        
        from collections import OrderedDict
        state_dict_remove_fc = OrderedDict()
        for x, y in torch.load(model_path).items():
            if x[:2] != 'fc':
                state_dict_remove_fc[x] = y
        model.load_state_dict(state_dict_remove_fc)

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        self.features = model
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features.forward(x)
        return output

# Assume input range is [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cuda')):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output

class Sphere20a(nn.Module):
    def __init__(self):
        super(Sphere20a, self).__init__()
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3, 64, 3, 2, 1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256, 256, 3, 1, 1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256, 256, 3, 1, 1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512 * 7 * 6, 512)


    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0), -1)
        x = self.fc5(x)

        return x

class Sphere20aFeatureExtractor(nn.Module):
    def __init__(self,
                 model_path,
                 feature_layer=28,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cuda')):
        super(Sphere20aFeatureExtractor, self).__init__()
        model = Sphere20a()
        
        from collections import OrderedDict
        state_dict_remove_fc = OrderedDict()
        for x, y in torch.load(model_path).items():
            if x[:3] != 'fc6':
                state_dict_remove_fc[x] = y
        model.load_state_dict(state_dict_remove_fc)

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        self.features = model
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features.forward(x)
        return output

# add Sphere10a
class Sphere10a(nn.Module):

    def __init__(self):
        super(Sphere10a, self).__init__()
        #input = B*3*28*24
        self.conv1_1 = nn.Conv2d(3, 64, 3, 2, 1) #=>B*64*14*12
        self.relu1_1 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1) #=>B*128*7*6
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_3 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1) #=>B*256*4*3
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1) #=>B*256*4*3
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_5 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1) #=>B*512*2*2
        self.relu4_1 = nn.PReLU(512)

        self.fc5 = nn.Linear(512 * 2 * 2, 512)
    
    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))

        x = self.relu4_1(self.conv4_1(x))

        x = x.view(x.size(0), -1)
        x = self.fc5(x)

        return x

class StyleMapping(nn.Module):

    def __init__(self, nb=8, style_dim=512):
        super(StyleMapping, self).__init__()

        self.mapping = [B.PixelNorm()]
        for _ in range(nb):
            self.mapping.append(nn.Linear(style_dim, style_dim))
            self.mapping.append(nn.LeakyReLU(0.2))
        
        self.mapping = nn.ModuleList(self.mapping)
    
    def forward(self, style):
        for block in self.mapping:
            style = block(style)
        return style