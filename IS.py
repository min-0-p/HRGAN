# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:35:36 2021

@author: MinYoung
"""


import os, sys, time

sys.path.append(os.getcwd())

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch, torchvision

import torch.nn.functional as F
import torchvision.transforms.functional as TF

from tqdm import tqdm

from model import Generator
from utils import one_hot_embedding
from dataset import CIFAR10_No_Label, CIFAR100_No_Label, GeneratorDataset

from pytorch_gan_metrics import get_inception_score

# Setting
DATASET_NAME        =    'CIFAR10'
# DATASET_NAME        =    'CIFAR100'
# HR_MULTIPLE         =            1  # High Resolution Multiplier
# HR_MULTIPLE         =            2  # High Resolution Multiplier
HR_MULTIPLE         =            4  # High Resolution Multiplier


# For Device
CUDA                =       torch.cuda.is_available()
DEVICE              =       torch.device('cuda' if CUDA else 'cpu')

_LOAD_PATH          =    'weights/' + DATASET_NAME + '/HRGAN_0250.pt'

BATCH_SIZE          =         100  # Generator batch size

# For Model
DIM_G               =     int(96)  # Generator dimensionality
DIM_Z               =     int(128)



if DATASET_NAME == 'CIFAR10':

    NUM_CLS = 10
    IMG_SIZE = 32
        
    tf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

if DATASET_NAME == 'CIFAR100':

    NUM_CLS = 100
    IMG_SIZE = 32
        
    tf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])    


gen = Generator(DIM_Z, DIM_G, DATASET_NAME, HR_MULTIPLE).to(DEVICE)

checkpoint = torch.load(_LOAD_PATH)
gen.load_state_dict(checkpoint['gen_dict'])
gen.eval()

start = time.time()
with torch.no_grad():
    
    if DATASET_NAME == 'CIFAR10':
        r_dset = CIFAR10_No_Label(f'../data/{DATASET_NAME}', transform= tf, download= True)
    elif DATASET_NAME == 'CIFAR100':
        r_dset = CIFAR100_No_Label(f'../data/{DATASET_NAME}', transform= tf, download= True)
        
    f_dset = GeneratorDataset(gen, DIM_Z, DATASET_NAME, BATCH_SIZE, NUM_CLS, DEVICE)
    
    # real_loader = torch.utils.data.DataLoader(r_dset, batch_size= BATCH_SIZE, shuffle= False, drop_last= False, num_workers= 0)
    fake_loader = torch.utils.data.DataLoader(f_dset, batch_size= BATCH_SIZE, shuffle= False, drop_last= False, num_workers= 0)
        
    # Get Inception Score from real dataset
    # r, rs = get_inception_score(real_loader)
    f, fs = get_inception_score(fake_loader)
    
    # torch.save([r, rs, f, fs],
    #            'is.pt')
        
    # print(f'REAL - Mean : {r:.2f} Std : {rs:.2f}')
    print(f'FAKE - Mean : {f:.2f} Std : {fs:.2f}')
    
print(f'Total Spent Time : \t\t\t{time.time() - start:.2f} seconds')