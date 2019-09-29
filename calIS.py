from models import networks
from data import create_dataloader, create_dataset
import argparse
import torch

if __name__ == '__main__':
    opt = {
            'gpu_ids': [0, 1, 2, 3],
            'network_F': {
                            'mode': 'Sphere20a',
                            'path': '/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/pretrained/sphere20a_20171020.pth',
                            'norm': 1
                        },
            'dataset': {
                            'name': 'CelebA',
                            'mode': 'SRHR',
                            'subset_file': None,
                            'phase': 'test',
                            'data_type': 'img',
                            'scale': 4,
                            'HR_size': 96,
                            'use_shuffle': False,
                            'batch_size': 1,
                            'n_workers': 1,
                            'color': False
                        }
        }

    parser = argparse.ArgumentParser()
    parser.add_argument('--HR_Root', type = str, default = "/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/VALHR", 
                        help = 'Path to val HR.')
    parser.add_argument('--SR_Root', type = str, default = "/GPUFS/nsccgz_yfdu_16/ouyry/SISRC/FaceSR-ESRGAN/dataset/CelebA/SR", 
                        help = 'Path to val SR.')
    parser.add_argument('--Norm', type = int, default = 1, help = 'Use Input Norm.')
    args = parser.parse_args()

    opt['dataset']['dataroot_SR'] = args.SR_Root
    opt['dataset']['dataroot_HR'] = args.HR_Root
    opt['network_F']['norm'] = args.Norm

    test_set = create_dataset(opt['dataset'])
    test_loader = create_dataloader(test_set, opt['dataset'])

    device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
    sphere = networks.define_F(opt).to(device)

    IS = 0
    idx = 0
    cos = torch.nn.CosineSimilarity()
    for data in test_loader:
        SR = data['SR'].to(device)
        HR = data['HR'].to(device)

        SR_vec = sphere(SR)
        HR_vec = sphere(HR)

        now_IS = cos(SR_vec, HR_vec)
        IS += now_IS
        idx += 1
    
    print("IS: %.4f"%(IS.item() / idx))