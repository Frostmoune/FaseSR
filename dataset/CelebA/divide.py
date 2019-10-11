import os, shutil

DIRS = ['Train_Align', 'Test_Align', 'Val_Align']

OUT_LR_PATH = 'LR'
OUT_HR_PATH = 'HR'
OUT_VAL_LR_PATH = 'VALLR'
OUT_VAL_HR_PATH = 'VALHR'

if __name__ == '__main__':
    try:
        os.makedirs(OUT_LR_PATH)
        os.makedirs(OUT_VAL_LR_PATH)
        os.makedirs(OUT_HR_PATH)
        os.makedirs(OUT_VAL_HR_PATH)
    except:
        pass

    divide_index = 200000
    img_names = [y for x in DIRS for y in os.listdir(x)]
    # hr_img_paths = [os.path.join(x, y) for x in DIRS for y in os.listdir(x)]
    lr_img_paths = [os.path.join(x + '_LR', y) for x in DIRS for y in os.listdir(x + '_LR')]

    for i, x in enumerate(img_names):
        if i < divide_index:
            # shutil.copy(hr_img_paths[i], os.path.join(OUT_HR_PATH, x))
            shutil.copy(lr_img_paths[i], os.path.join(OUT_LR_PATH, x))
        else:
            # shutil.copy(hr_img_paths[i], os.path.join(OUT_VAL_HR_PATH, x))
            shutil.copy(lr_img_paths[i], os.path.join(OUT_VAL_LR_PATH, x))
        
        print(i)