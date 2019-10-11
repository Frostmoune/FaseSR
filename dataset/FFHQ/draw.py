import json
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
    Image_Root = '/GPUFS/nsccgz_yfdu_16/ouyry/images1024x1024/'
    LR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/FFHQ/LR/'
    HR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/FFHQ/HR/'

    try:
        os.makedirs(LR)
        os.makedirs(HR)
    except:
        pass
    
    landmarks = {}
    with open('ffhq-dataset-v1.json', 'r') as f:
        landmarks = json.load(f)
    
    for i in range(70000):
        index = str(i)
        img_name = landmarks[index]['image']['file_path'][-9:]
        print(i, img_name)
        now_road = Image_Root + img_name
        now_landmark = landmarks[index]['image']['face_landmarks']

        if os.path.exists(now_road):
            img = cv2.imread(now_road)
        else:
            continue

        left_eyes_x, left_eyes_y = 0, 0
        right_eyes_x, right_eyes_y = 0, 0
        now_crop_landmark = []
        for j, [x, y] in enumerate(now_landmark):
            if j in [36, 37, 38, 39, 40, 41]:
                left_eyes_x += x
                left_eyes_y += y
            if j in [42, 43, 44, 45, 46, 47]:
                right_eyes_x += x
                right_eyes_y += y
            if j in [30, 48, 54]:
                now_crop_landmark.append([int(x), int(y)])

        left_eyes_x /= 6
        left_eyes_y /= 6
        right_eyes_x /= 6
        right_eyes_y /= 6
        cv2.circle(img, (int(left_eyes_x), int(left_eyes_y)), 2, (0, 0, 255), -1)
        cv2.circle(img, (int(right_eyes_x), int(right_eyes_y)), 2, (0, 0, 255), -1)

        now_crop_landmark = [[int(left_eyes_x), int(left_eyes_y)]] + [[int(right_eyes_x), int(right_eyes_y)]] + now_crop_landmark

        save_load = HR + img_name
        img = alignment(img, now_crop_landmark)
        cv2.imwrite(save_load, img)