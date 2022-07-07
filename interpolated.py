# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:52:53 2021

@author: MinYoung
"""


import os, sys

sys.path.append(os.getcwd())

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch, torchvision
from model import Generator
from utils import one_hot_embedding


# Dataset Choice
DATASET_NAME        =    'CIFAR10'

# Path
_LOAD_PATH          =       'weights/' + DATASET_NAME + '/HRGAN_0250.pt'

# For Device
CUDA                =       torch.cuda.is_available()
DEVICE              =       torch.device('cuda' if CUDA else 'cpu')

# Constant
NUM_CIFAR10         =        50000

# H Params for Training
BATCH_SIZE          =           64  # Generator batch size

# For Model
DIM_G               =     int(96)  # Generator dimensionality
DIM_D               =     int(128)  # Critic dimensionality
DIM_Z               =     int(128)
HR_MULTIPLE         =            4  # High Resolution Multiplier


# ---------------------------------------- DATASET-------------------------------------------------
if DATASET_NAME == 'CIFAR10':

    NUM_CLS = 10
    IMG_SIZE = 32
    
    CRITIC_NUM_RESIDUAL = 2
    
# ----------------------------------------------------- MODEL --------------------------------------------------
gen = Generator(DIM_Z, DIM_G, DATASET_NAME, HR_MULTIPLE).to(DEVICE)
gen.eval()


# --------------------------------------------------- LOADING -------------------------------------------------

checkpoint = torch.load(_LOAD_PATH)
gen.load_state_dict(checkpoint['gen_dict'])

# ------------------------------------------ TESTING ----------------------------------------------------

with torch.no_grad():
        
    for batch_idx in range(10):
    
        noise_list = []
        
        for random in range(2 * DIM_Z):
            
            if random == DIM_Z:
                noise_a = torch.cat(noise_list, dim= 0).to(DEVICE).view(1, -1)
                noise_list = []
                
            while(True):
                noise = torch.randn(1)
                if -0.5 < noise < 0.5:
                    noise_list.append(noise)
                    break
            
        noise_b = torch.cat(noise_list, dim= 0).to(DEVICE).view(1, -1)
        
        
        diff = (noise_b.data - noise_a.data) / 9
        
        noise_list = []
        for i in range(1, 9):
            noise_list.append(noise_a.data + i * diff.data )
                   
        
        noise = torch.cat([noise_a] + noise_list + [noise_b], dim= 0)
        noise = noise.repeat(10, 1)
                        
        label = one_hot_embedding(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10).reshape(10, 10).permute(1, 0).reshape(-1).to(DEVICE).contiguous(), 10, device= DEVICE)
        
        _, fake = gen(noise, label)
        import torchvision.transforms.functional as TF
        fake = TF.resize(fake, (64,64))
        
        # Save Images generated with fixed noises
        img_grid_fake = torchvision.utils.make_grid(fake[:100], normalize=True, nrow = 10)
        torchvision.utils.save_image(img_grid_fake, './images/test/img_' + f'{batch_idx:04d}' + '.png')