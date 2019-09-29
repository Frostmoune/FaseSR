import cv2
import numpy as np
from numpy.random import normal
from skimage import io
from scipy import ndimage
import os
from os.path import join
    
    
class PostProcessor():
    def __init__(self, src_path, dest_path):
        self.src = src_path
        self.dest = dest_path
        if not os.path.isdir(self.dest):
            os.mkdir(self.dest)
        
        self.filelist = []
        for f in os.listdir(self.src):
            if f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.jpg'):
                self.filelist.append(f)
        
    def filter_3x3(self, kernel=None):
        if kernel is None:
            kernel = np.array([
                [0, -.1, 0],
                [-.1, 1.4, -.1],
                [0, -.1, 0]
            ], np.float32)
        for f in self.filelist:
            img = cv2.imread(join(self.src, f))
            img2 = cv2.filter2D(img, -1, kernel = kernel)
            self.save(img2, f)


    def filter_5x5(self, kernel=None):
        if kernel is None:
            kernel = np.array([
                [0, 0, -.01, 0, 0],
                [0, -.01, -.08, -.01, 0],
                [-.01, -.08, 1.4, -.08, -.01],
                [0, -.01, -.08, -.01, 0],
                [0, 0, -.01, 0, 0]
            ])
        for f in self.filelist:
            img = cv2.imread(join(self.src, f))
            img2 = cv2.filter2D(img, -1, kernel = kernel)
            self.save(img2, f)
    
    def save(self, img, name):
        cv2.imwrite(join(self.dest, name), img)
        

# __main__ is just for test        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='No description')
    parser.add_argument('--src', type=str, help='source file')
    parser.add_argument('--dest', type=str, help='destination file')
    opt = parser.parse_args()
    src_path = opt.src
    dest_path = opt.dest_path


    # Use like this:
    p = PostProcessor(src_path, dest_path)
    p.filter_3x3()