import cv2
from matlab_cp2tform import get_similarity_transform_for_cv2
import numpy as np
import os

def alignment(src_img, src_pts):
    of = 0
    ref_pts = [ [30.2946+of, 51.6963+of],[65.5318+of, 51.5014+of],
        [48.0252+of, 71.7366+of],[33.5493+of, 92.3655+of],[62.7299+of, 92.2041+of] ]
    crop_size = (96+of*2, 112+of*2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

def generateAlignFace(img_dir, save_dir, img_to_landmarks, label = 'Train'):
    for x in os.listdir(img_dir):
        img_path = os.path.join(img_dir, x)
        img = cv2.imread(img_path)

        src_pts = img_to_landmarks[x]
        img = alignment(img, src_pts)
        
        save_path = os.path.join(save_dir, x)
        cv2.imwrite(save_path, img)

        print("%s: %s"%(label, x))

if __name__ == '__main__':
    TRAIN_IMG_DIR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Train'
    VAL_IMG_DIR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Val'
    TEST_IMG_DIR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Test'
    LANDMARKS_DIR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/list_landmarks_align_celeba.txt'

    TRAIN_ALIGN_DIR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Train_Align'
    VAL_ALIGN_DIR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Val_Align'
    TEST_ALIGN_DIR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Test_Align'

    try:
        # os.makedirs(TRAIN_ALIGN_DIR)
        # os.makedirs(TEST_ALIGN_DIR)
        os.makedirs(VAL_ALIGN_DIR)
    except:
        pass

    img_to_landmarks = {}
    with open(LANDMARKS_DIR, 'r') as f:
        for line in f.readlines():
            info = line.strip('\n').split(' ')
            info = list(filter(lambda x: x != '', info))
            
            if info[0][-3:] == 'jpg':
                img_to_landmarks[info[0]] = []
                i = 1
                while i + 1 < len(info):
                    img_to_landmarks[info[0]].append([int(info[i]), int(info[i + 1])])
                    i += 2
    
    # generateAlignFace(TRAIN_IMG_DIR, TRAIN_ALIGN_DIR, img_to_landmarks, label = 'Train')
    generateAlignFace(VAL_IMG_DIR, VAL_ALIGN_DIR, img_to_landmarks, label = 'Val')
    # generateAlignFace(TEST_IMG_DIR, TEST_ALIGN_DIR, img_to_landmarks, label = 'Test')