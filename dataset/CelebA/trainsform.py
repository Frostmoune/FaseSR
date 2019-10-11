import os, shutil, cv2
import numpy as np
import argparse

def randomTransform(img):
    x, y, c = img.shape
    new_img = img + 1.5 * np.random.randn(x, y, c)
    new_img = new_img.astype(np.uint8)
    return new_img

def sharp(img):
    kernel = np.array([[0, -0.05, 0], [-0.05, 1.2, -0.05], [0, -0.05, 0]], np.float32)
    new_img = cv2.filter2D(img, -1, kernel = kernel)
    return new_img

def average(img):
    new_img = cv2.blur(img, (3, 3))
    return new_img

def gaussian(img):
    new_img = cv2.GaussianBlur(img, (3, 3), 0)
    return new_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--HR_Root', type = str, default = "/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/VALHR", 
                        help = 'Path to val HR.')
    parser.add_argument('--Save_Root', type = str, default = "/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/VALHR_Transform", 
                        help = 'Path to new HR.')
    parser.add_argument('--Transform', type = str, default = "random", help = 'Type of transform')
    args = parser.parse_args()

    HR_Root = args.HR_Root
    Save_Root = args.Save_Root
    trans = args.Transform

    try:
        os.makedirs(Save_Root)
    except:
        pass
    
    for i, hr_name in enumerate(os.listdir(HR_Root)):
        hr_path = os.path.join(HR_Root, hr_name)
        img = cv2.imread(hr_path)

        if trans == 'random':
            new_img = randomTransform(img)
        elif trans == 'sharp':
            new_img = sharp(img)
        elif trans == 'average':
            new_img = average(img)
        elif trans == 'gaussian':
            new_img = gaussian(img)
            
        save_path = os.path.join(Save_Root, hr_name)
        cv2.imwrite(save_path, new_img)
        print(i)