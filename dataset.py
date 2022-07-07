# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:21:23 2021

@author: MinYoung
"""

import torch, torchvision

import random

from PIL import Image
from tqdm import tqdm

from torchvision.transforms.functional import resize
from utils import one_hot_embedding


class CIFAR10_No_Label(torchvision.datasets.CIFAR10):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            image : Int
        """
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img
    
class CIFAR100_No_Label(torchvision.datasets.CIFAR100):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            image : Int
        """
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img
        
    

class GeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, G, dim_z, dataset_name, batch_size, num_cls, device):
        
        if dataset_name == 'CIFAR10':
            labels = list(range(10)) * 5000 
        elif dataset_name == 'CIFAR100':
            labels = list(range(100)) * 500
        
        
        random.shuffle(labels)
        
        fake_list = []        
        
        with torch.no_grad():
            
            loop = tqdm(range(0, len(labels), batch_size))
            labels = torch.tensor(labels).to(torch.long).to(device)
            loop.set_description(f'Generating Fake Samples')
            
            for i in loop:
                label = labels[i : i + batch_size]
                label = one_hot_embedding(label, num_cls, device=device)
                noise = torch.randn(batch_size, dim_z).to(device)
                _, fake = G(noise, label)
                fake = (fake + 1.0) / 2.0
                fake_list.append(fake.cpu())
                
            self.fakes = torch.cat(fake_list, dim= 0)
            
    def __len__(self):
        return self.fakes.size(0)
    
    def __getitem__(self, index):
        return self.fakes[index]
