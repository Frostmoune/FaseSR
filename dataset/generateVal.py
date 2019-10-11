import os
import shutil
import random

if __name__ == '__main__':
    LR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/LR/'
    HR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/HR/'
    TRAINLR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/TRAINLR/'
    TRAINHR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/TRAINHR/'
    VALLR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/VALLR/'
    VALHR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/VALHR/'

    try:
        os.makedirs(TRAINLR)
        os.makedirs(TRAINHR)
        os.makedirs(VALLR)
        os.makedirs(VALHR)
    except:
        pass

    tHR = os.listdir(HR)
    random.shuffle(tHR)
    vHR = tHR[270000:]
    tHR = tHR[:270000]
    
    print("Train")
    for i, img_name in enumerate(tHR):
        shutil.copy(LR + img_name, TRAINLR)
        shutil.copy(HR + img_name, TRAINHR)
        print(i)
    
    print("Val")
    for i, img_name in enumerate(vHR):
        shutil.copy(LR + img_name, VALLR)
        shutil.copy(HR + img_name, VALHR)
        print(i)
