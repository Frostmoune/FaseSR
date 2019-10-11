import os
import time
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import numpy as np
import cv2

from model import Sphere

class CelebADataset(Dataset):
    """Dog breed identification dataset."""

    def __init__(self, img_dir, label_dir, transform = None, is_train = 1):
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        
        self.img_to_labels = {}
        self.img_list = []
        with open(label_dir, 'r') as f:
            for line in f.readlines():
                info = line.strip('\n').split(' ')
                img_path = os.path.join(self.img_dir, info[0])
                
                if os.path.exists(img_path):
                    self.img_list.append(info[0])
                    self.img_to_labels[info[0]] = int(info[1]) - 1
            random.shuffle(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if random.random() > 0.5 and self.is_train: 
            image = cv2.flip(image, 1)
            
        image = image.transpose((2, 0, 1))
        image = (image - 127.5) / 128.0
        image = torch.from_numpy(image).float()
        # image = Image.open(img_path)
        
        label = self.img_to_labels[self.img_list[idx]]

        if self.transform:
            image = self.transform(image)

        return image, label

TRAIN_IMG_DIR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Train'
TRAIN_LABEL_DIR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Train_CelebA.txt'
VAL_IMG_DIR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Val'
VAL_LABEL_DIR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Val_CelebA.txt'
TEST_IMG_DIR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Test'
TEST_LABEL_DIR = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/Test_CelebA.txt'

if __name__ == '__main__':
    epochs = 100
    batch_size = 80
    lr = 2e-4
    lr_steps = [45340, 136020]
    val_iter = 500

    train_state_dir = 'train_state4'
    model_dir = 'model4'
    try:
        os.makedirs(train_state_dir)
        os.makedirs(model_dir)
    except:
        pass
    
    train_data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-5, 5)),
        transforms.RandomAffine(degrees = 0, translate = (0.03, 0.02)),
        transforms.ToTensor()
    ])
    test_data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = CelebADataset(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, transform = None, is_train = 1)
    test_dataset = CelebADataset(TEST_IMG_DIR, TEST_LABEL_DIR, transform = None, is_train = 0)
    val_dataset = CelebADataset(VAL_IMG_DIR, VAL_LABEL_DIR, transform = None, is_train = 0)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, drop_last = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, drop_last = False)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = 8, drop_last = False)
    per_iters = len(train_dataset) // batch_size + 1

    my_sphere = Sphere(lr, lr_steps)
    epoch, now_iter = 0, 1
    # epoch, now_iter = my_sphere.resume_training('/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/Sphere/train_state2/137000.state')
    # my_sphere.load_model('/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/Sphere/model2/Sphere_iter137000.pth')
    try:
        while epoch <= epochs:
            correct, loss = 0, 0
            for data in train_loader:
                imgs, labels = data
                my_sphere.update_lr()
                now_loss, now_correct = my_sphere.feed_data(imgs, labels)
                my_sphere.optimize_parameters()
                
                correct += now_correct
                loss += now_loss
                
                if now_iter % 100 == 0:
                    print("Train:<Epoch: %d, Iter: %d, Loss: %.4f, Correct: %d/%d>"%(epoch, now_iter, now_loss, now_correct, batch_size))
                    
                if now_iter % val_iter == 0:
                    val_correct, val_loss = 0, 0
                    idx = 0
                    for data in val_loader:
                        vimgs, vlabels = data
                        vnow_loss, vnow_correct = my_sphere.feed_data(vimgs, vlabels)

                        val_correct += vnow_correct
                        val_loss += vnow_loss
                        idx += 1

                    val_loss /= idx
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end = " ")
                    print("Val:<Epoch: %d, Iter: %d, Loss: %.4f, Correct: %d/%d>"%(epoch, now_iter, val_loss, val_correct, len(val_dataset)))

                    my_sphere.save_model(model_dir, now_iter)
                    my_sphere.save_training_state(epoch, now_iter, train_state_dir)
                    
                now_iter += 1
            
            loss /= per_iters
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), end = " ")
            print("Train:<Epoch: %d, Loss: %.4f, Correct: %d/%d>"%(epoch, loss, correct, len(train_dataset)))
            
            epoch += 1

    except KeyboardInterrupt:
        my_sphere.save_model(model_dir, now_iter)
        my_sphere.save_training_state(epoch, now_iter, train_state_dir)