import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util


class SRHRDataset(data.Dataset):
    '''
    Read SR and HR image pairs.
    If only HR image is provided, generate SR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(SRHRDataset, self).__init__()
        self.opt = opt
        self.paths_SR = None
        self.paths_HR = None
        self.SR_env = None  # environment for lmdb
        self.HR_env = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) \
                        for line in f])
            if opt['dataroot_SR'] is not None:
                raise NotImplementedError('Now subset only supports generating SR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.HR_env, self.paths_HR = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])
            self.SR_env, self.paths_SR = util.get_image_paths(opt['data_type'], opt['dataroot_SR'])

        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_SR and self.paths_HR:
            assert len(self.paths_SR) == len(self.paths_HR), \
                'HR and SR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_SR), len(self.paths_HR))

        self.random_scale_list = [1]

    def __getitem__(self, index):
        HR_path, SR_path = None, None
        scale = self.opt['scale']
        HR_size = self.opt['HR_size']

        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = util.read_img(self.HR_env, HR_path)
        # modcrop in the validation / test phase
        # change color space if necessary
        if self.opt['color']:
            img_HR = util.channel_convert(img_HR.shape[2], self.opt['color'], [img_HR])[0]

        # get SR image
        if self.paths_SR:
            SR_path = self.paths_SR[index]
            img_SR = util.read_img(self.SR_env, SR_path)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_HR.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, HR_size)
                W_s = _mod(W_s, random_scale, scale, HR_size)
                img_HR = cv2.resize(np.copy(img_HR), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_HR.ndim == 2:
                    img_HR = cv2.cvtColor(img_HR, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_HR.shape
            # using matlab imresize
            img_SR = util.imresize_np(img_HR, 1 / scale, True)
            if img_SR.ndim == 2:
                img_SR = np.expand_dims(img_SR, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_HR.shape
            # if H < HR_size or W < HR_size:
            #     img_HR = cv2.resize(
            #         np.copy(img_HR), (HR_size, HR_size), interpolation=cv2.INTER_LINEAR)
            #     # using matlab imresize
            #     img_SR = util.imresize_np(img_HR, 1 / scale, True)
            #     if img_SR.ndim == 2:
            #         img_SR = np.expand_dims(img_SR, axis=2)

            # H, W, C = img_SR.shape
            # SR_size = HR_size // scale

            # # randomly crop
            # rnd_h = random.randint(0, max(0, H - SR_size))
            # rnd_w = random.randint(0, max(0, W - SR_size))
            # img_SR = img_SR[rnd_h:rnd_h + SR_size, rnd_w:rnd_w + SR_size, :]
            # rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            # img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]

            # augmentation - flip, rotate
            img_SR, img_HR = util.augment([img_SR, img_HR], self.opt['use_flip'], \
                self.opt['use_rot'])

        # change color space if necessary
        if self.opt['color']:
            img_SR = util.channel_convert(C, self.opt['color'], [img_SR])[0] # TODO during val no definetion

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_SR = img_SR[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_SR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_SR, (2, 0, 1)))).float()

        if SR_path is None:
            SR_path = HR_path
        return {'SR': img_SR, 'HR': img_HR, 'SR_path': SR_path, 'HR_path': HR_path}

    def __len__(self):
        return len(self.paths_HR)
