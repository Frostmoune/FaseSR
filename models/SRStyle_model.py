import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')


class SRStyleModel(BaseModel):
    def __init__(self, opt):
        super(SRStyleModel, self).__init__(opt)
        train_opt = opt['train']

        self.with_style = opt["network_G"]['with_style']
        self.with_latent = train_opt['with_latent']

        self.netG = networks.define_G(opt).to(self.device)
        if self.with_style:
            self.netS = networks.define_S(opt, use_bn=False).to(self.device)
        else:
            self.style = None

        self.load()

        if self.is_train:
            self.netG.train()

            if self.with_style:
                self.netS.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(
                optim_params, lr=train_opt['lr_G'], weight_decay=wd_G)
            self.optimizers.append(self.optimizer_G)

            # S
            if self.with_style:
                wd_S = train_opt['weight_decay_S'] if train_opt['weight_decay_S'] else 0
                self.optimizer_S = torch.optim.Adam(self.netS.parameters(), lr=train_opt['lr_S'], \
                    weight_decay=wd_S, betas=(train_opt['beta1_S'], 0.999))
                self.optimizers.append(self.optimizer_S)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        self.var_L = data['LR'].to(self.device)  # LR
        if need_HR:
            self.real_H = data['HR'].to(self.device)  # HR
            if self.with_style:
                if self.with_latent:
                    batch = data['LR'].shape[0]
                    latent = torch.randn(batch, 512).to(self.device)
                    self.style = self.netS(latent)
                else:
                    self.style = self.netS(self.var_L)

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        if self.with_style:
            self.optimizer_S.zero_grad()

        self.fake_H = self.netG(self.var_L, self.style)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()

        self.optimizer_G.step()

        if self.with_style:
            self.optimizer_S.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        if self.with_style:
            self.netS.eval()

        with torch.no_grad():
            self.fake_H = self.netG(self.var_L, self.style)

        self.netG.train()
        if self.with_style:
            self.netS.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()
        for k, v in self.netG.named_parameters():
            v.requires_grad = False

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)

        for k, v in self.netG.named_parameters():
            v.requires_grad = True
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                            self.netG.module.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

        if self.with_style:
            s, n = self.get_network_description(self.netS)
            net_struc_str = '{} - {}'.format(self.netS.__class__.__name__,
                                            self.netS.module.__class__.__name__)

            logger.info('Network S structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)
        
        if self.with_style:
            load_path_S = self.opt['path']['pretrain_model_S']
            if load_path_S is not None:
                logger.info('Loading pretrained model for S [{:s}] ...'.format(load_path_S))
                self.load_network(load_path_S, self.netS)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)

        if self.with_style:
            self.save_network(self.netS, 'S', iter_step)