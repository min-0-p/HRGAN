# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:15:12 2021

@author: MinYoung
"""

import os, sys

sys.path.append(os.getcwd())

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch, torchvision
from model import Generator


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
fixed_label = checkpoint['fixed_label']

# ------------------------------------------ TESTING ----------------------------------------------------

with torch.no_grad():
        
    for batch_idx in range(10):
    
        noise = torch.randn(100, DIM_Z).to(DEVICE)
        _, fake = gen(noise, fixed_label)
        
        # Save Images generated with fixed noises
        img_grid_fake = torchvision.utils.make_grid(fake[:100], normalize=True, nrow = 10)
        torchvision.utils.save_image(img_grid_fake, './images/test/img_' + f'{batch_idx:04d}' + '.png')
        
                    

        
        
        

