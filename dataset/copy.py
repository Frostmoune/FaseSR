import os
import shutil

if __name__ == '__main__':
    src = ['CelebA/VALHR', 'CelebA/VALLR']
    dst = ['HR', 'LR']
    
    for i, root in enumerate(src):
        try:
            os.mkdir(dst[i])
        except:
            pass
        for x in os.listdir(root):
            now_path = os.path.join(root, x)
            shutil.copy(now_path, dst[i])
            print(i, root, x)