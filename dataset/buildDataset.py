import os, shutil, cv2
import torch
import numpy as np
from GEN_LR import reShape, saveImage

OUT_TRAIN_PATH = 'CelebA/Train/'
OUT_VAL_PATH = 'CelebA/Val/'
OUT_TEST_PATH = 'CelebA/Test/'

def readInfo(in_file):
    out_dict = {}
    with open(in_file, 'r') as f:
        for line in f.readlines():
            info = line.strip('\n').split(' ')
            out_dict[info[0]] = info[1]
    return out_dict

if __name__ == '__main__':
    try:
        os.makedirs(OUT_TRAIN_PATH)
        os.makedirs(OUT_VAL_PATH)
        os.makedirs(OUT_TEST_PATH)
    except:
        pass

    size_ = (176, 216)

    train_pic_to_id = readInfo('CelebA/Train_CelebA.txt')
    val_pic_to_id = readInfo('CelebA/Val_CelebA.txt')
    test_pic_to_id = readInfo('CelebA/Test_CelebA.txt')

    for i in range(1, 202600):
        img_name = '%06d.jpg'%(i)
        in_file = 'img_align_celeba/' + img_name
        pic = cv2.imread(in_file)

        if os.path.exists(in_file) and (not (pic is None)):
            img = reShape(pic, size_ = size_)
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()

            if img_name in train_pic_to_id:
                saveImage(img, OUT_TRAIN_PATH + img_name)
            elif img_name in val_pic_to_id: 
                saveImage(img, OUT_VAL_PATH + img_name)
            elif img_name in test_pic_to_id:
                saveImage(img, OUT_TEST_PATH + img_name)

        print(i)
            