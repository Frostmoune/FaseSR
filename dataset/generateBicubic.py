import os, shutil, cv2
import torch
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--LR_Root', type = str, default = "/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/VALLR", 
                        help = 'Path to val LR.')
    parser.add_argument('--SR_Root', type = str, default = "/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/VALSR_bicubic", 
                        help = 'Path to val SR.')
    args = parser.parse_args()

    LR_Root = args.LR_Root
    SR_Root = args.SR_Root

    try:
        os.makedirs(SR_Root)
    except:
        pass
    
    for i, lr_name in enumerate(os.listdir(LR_Root)):
        lr_path = os.path.join(LR_Root, lr_name)
        img = cv2.imread(lr_path)
        img = cv2.resize(img, (96, 112), interpolation = cv2.INTER_CUBIC)

        sr_path = os.path.join(SR_Root, lr_name)
        cv2.imwrite(sr_path, img)
        print(i)