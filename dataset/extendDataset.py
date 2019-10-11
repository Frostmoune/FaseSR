import os, shutil
from GEN_LR import downSample, saveImage

PATHS = ['img_align_celeba']

OUT_LR_PATH = 'TrainData1/VALLR/'
OUT_HR_PATH = 'TrainData1/VALHR/'

if __name__ == '__main__':
    try:
        os.makedirs(OUT_HR_PATH)
        os.makedirs(OUT_LR_PATH)
    except:
        pass

    i = 0
    size_ = (176, 216)
    for path in PATHS:
        if os.path.exists(path):
            for img_name in os.listdir(path):
                in_file = path + '/' + img_name
                save_name = '%06d.jpg'%(i)

                if i >= 190000:
                    if os.path.exists('TrainData1/LR/' + save_name):
                        os.remove('TrainData1/LR/' + save_name)
                        os.remove('TrainData1/HR/' + save_name)

                    pic, rlt = downSample(in_file, size_ = size_)

                    if not (pic is None):
                        saveImage(rlt, OUT_LR_PATH + save_name)
                        saveImage(pic, OUT_HR_PATH + save_name)

                    print(i)
                    
                i += 1