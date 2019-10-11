import os, shutil, cv2
import torch
import numpy as np
from GEN_LR import downSample, saveImage
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR_Root', type = str, default = "/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Train_Align", 
                        help = 'Path to val HR.')
    parser.add_argument('--LR_Root', type = str, default = "/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Train_Align_LR", 
                        help = 'Path to val SR.')
    args = parser.parse_args()

    HR_Root = args.HR_Root
    LR_Root = args.LR_Root

    try:
        os.makedirs(LR_Root)
    except:
        pass
    
    for i, hr_name in enumerate(os.listdir(HR_Root)):
        hr_path = os.path.join(HR_Root, hr_name)
        _, rlt = downSample(hr_path)

        lr_path = os.path.join(LR_Root, hr_name)
        saveImage(rlt, lr_path)
        print(i)