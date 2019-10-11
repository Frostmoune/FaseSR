import os, shutil
import random
import numpy as np

DIRS = ['Train_Align', 'Test_Align', 'Val_Align']

OUT_LR_PATH = 'LR_Small'
OUT_HR_PATH = 'HR_Small'
OUT_VAL_LR_PATH = 'VALLR_Small'
OUT_VAL_HR_PATH = 'VALHR_Small'

if __name__ == '__main__':
    try:
        os.makedirs(OUT_LR_PATH)
        os.makedirs(OUT_VAL_LR_PATH)
        os.makedirs(OUT_HR_PATH)
        os.makedirs(OUT_VAL_HR_PATH)
    except:
        pass

    img_names = np.array([y for x in DIRS for y in os.listdir(x)])
    hr_img_paths = np.array([os.path.join(x, y) for x in DIRS for y in os.listdir(x)])
    lr_img_paths = np.array([os.path.join(x + '_LR', y) for x in DIRS for y in os.listdir(x + '_LR')])

    total_len = 10100
    divide_index = 10000
    indexs = list(range(0, len(img_names)))
    random.shuffle(indexs)
    indexs = indexs[:total_len]

    img_names = img_names[indexs].tolist()
    hr_img_paths = hr_img_paths[indexs].tolist()
    lr_img_paths = lr_img_paths[indexs].tolist()

    for i, x in enumerate(img_names):
        if i < divide_index:
            shutil.copy(hr_img_paths[i], os.path.join(OUT_HR_PATH, x))
            shutil.copy(lr_img_paths[i], os.path.join(OUT_LR_PATH, x))
        else:
            shutil.copy(hr_img_paths[i], os.path.join(OUT_VAL_HR_PATH, x))
            shutil.copy(lr_img_paths[i], os.path.join(OUT_VAL_LR_PATH, x))
        
        print(i)
