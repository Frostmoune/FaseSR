import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from models.base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss
from torch.autograd import Variable
logger = logging.getLogger('base')

class SR2HRModel(BaseModel):
    def __init__(self, opt):
        super(SR2HRModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)  # G
        self.with_latent = False
            
        if self.is_train:
            # forward with style
            self.with_latent = train_opt['with_latent']
            self.netG.train()
            if self.with_latent:
                self.netS = networks.define_S(opt).to(self.device)  # S
                self.netS.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                elif l_fea_type == 'cos':
                    self.cri_fea = nn.CosineEmbeddingLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            # self.optimizer_G = torch.optim.SGD(optim_params, lr=train_opt['lr_G'])
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)

            # S
            if self.with_latent:
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
        # LR
        self.var_S = data['SR'].to(self.device)

        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)
            if self.with_latent:
                batch = data['SR'].shape[0]
                latent = torch.rand(batch, 512).to(self.device)
                self.style = self.netS(latent)
            else:
                self.style = None

            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        self.fake_H = self.netG(self.var_S, self.style)
        real_fea = self.netF(self.var_H).detach()
        fake_fea = self.netF(self.fake_H)
        
        self.optimizer_G.zero_grad()
        if self.with_latent:
            self.optimizer_S.zero_grad()
        l_g_total = 0
        batch_size = fake_fea.shape[0]

        if self.cri_pix:  # pixel loss
            l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
            l_g_total += l_g_pix
        if self.cri_fea:  # feature loss
            if isinstance(self.cri_fea, nn.CosineEmbeddingLoss):
                flags = Variable(torch.ones(batch_size)).to(self.device)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea, flags)
            else:
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
            l_g_total += l_g_fea

        l_g_total.backward()
        self.optimizer_G.step()
        if self.with_latent:
            self.optimizer_S.step()

        # G
        if self.cri_pix:
            self.log_dict['l_g_pix'] = l_g_pix.item()
        if self.cri_fea:
            self.log_dict['l_g_fea'] = l_g_fea.item()

        #total
        self.log_dict['l_g_total'] = l_g_total.item()

    def test(self):
        self.netG.eval()
        if self.with_latent:
            self.netS.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_S, self.style)
        self.netG.train()
        if self.with_latent:
            self.netS.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True, need_FeatureLoss=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_S.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
            if need_FeatureLoss:
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                if isinstance(self.cri_fea, nn.CosineEmbeddingLoss):
                    batch_size = fake_fea.shape[0]
                    flags = Variable(torch.ones(batch_size)).to(self.device)
                    out_dict['Feature_Loss'] = self.cri_fea(fake_fea, real_fea, flags)
                else:
                    out_dict['Feature_Loss'] = self.cri_fea(fake_fea, real_fea)
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                            self.netG.module.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            if self.with_latent:
                s, n = self.get_network_description(self.netS)
                net_struc_str = '{} - {}'.format(self.netS.__class__.__name__,
                                                self.netS.module.__class__.__name__)

                logger.info('Network S structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                self.netF.module.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

        load_path_S = self.opt['path']['pretrain_model_S']
        if self.with_latent and self.opt['is_train'] and load_path_S is not None:
            logger.info('Loading pretrained model for S [{:s}] ...'.format(load_path_S))
            self.load_network(load_path_S, self.netS)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        if self.with_latent:
            self.save_network(self.netS, 'S', iter_step)