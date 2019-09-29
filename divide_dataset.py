from models import networks
from data import create_dataloader, create_dataset
import argparse
import torch
from sklearn.cluster import MiniBatchKMeans
import os
import numpy as np
import shutil
import pickle

if __name__ == '__main__':
    opt = {
            'gpu_ids': [0, 1, 2, 3],
            'network_F': {
                            'mode': 'Sphere20a',
                            'path': '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/pretrained/sphere20a_20171020.pth'
                        },
            'dataset': {
                            'name': 'CelebA',
                            'mode': 'LRHR',
                            'subset_file': None,
                            'phase': 'train',
                            'data_type': 'img',
                            'scale': 4,
                            'HR_size': 96,
                            'use_shuffle': True,
                            'use_flip': False,
                            'use_rot': False,
                            'batch_size': 40,
                            'n_workers': 4,
                            'color': False
                        }
        }

    parser = argparse.ArgumentParser()
    parser.add_argument('--HR_Root', type = str, default = "/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/FFHQ/HR", 
                        help = 'Path to val HR.')
    parser.add_argument('--LR_Root', type = str, default = "/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/FFHQ/LR", 
                        help = 'Path to val LR.')
    parser.add_argument('--Clusters', type = int, default = 3, help = 'Number of clusters')
    parser.add_argument('--Train', type = int, default = 0, help = 'Train or not')
    parser.add_argument('--Model_Path', type = str, default = "/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/FFHQ/cluster.model", 
                        help = 'Path to Cluster model')
    args = parser.parse_args()

    Root = '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/FFHQ'
    opt['dataset']['dataroot_LR'] = args.LR_Root
    opt['dataset']['dataroot_HR'] = args.HR_Root

    test_set = create_dataset(opt['dataset'])
    test_loader = create_dataloader(test_set, opt['dataset'])

    device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
    sphere = networks.define_F(opt).to(device)

    for i in range(args.Clusters):
        try:
            os.makedirs(args.HR_Root + str(i))
            os.makedirs(args.LR_Root + str(i))
        except:
            pass

    vectors = None
    LR_paths = []
    HR_paths = []
    for i, data in enumerate(test_loader):
        HR = data['HR'].to(device)
        HR_vec = sphere(HR).to('cpu').numpy()
        if vectors is None:
            vectors = HR_vec
        else:
            vectors = np.concatenate((vectors, HR_vec), axis = 0)
        LR_paths += data['LR_path']
        HR_paths += data['HR_path']
        print("Sphere %d batch"%i)
    print(vectors.shape)
    print("Sphere Done ...")

    mean = np.mean(vectors, axis = 0, keepdims = True)
    std = np.std(vectors, axis = 0, keepdims = True)
    vectors = (vectors - mean) / std
    if args.Train:
        model = MiniBatchKMeans(n_clusters = args.Clusters, batch_size = 2000, random_state = 0, max_iter = 5000)
        for i in range(0, vectors.shape[0], 2000):
            model.partial_fit(vectors[i:i+2000, :])
        with open(args.Model_Path, 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(args.Model_Path, 'rb') as f:
            model = pickle.load(f)
    labels = model.predict(vectors)
    print("Cluster Done ...")

    for i, label in enumerate(labels):
        print(i)
        shutil.copy(LR_paths[i], args.LR_Root + str(label))
        shutil.copy(HR_paths[i], args.HR_Root + str(label))
    print("Done")