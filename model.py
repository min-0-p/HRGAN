# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 19:07:23 2021

@author: MinYoung
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

from utils import CondBatchNorm2d, init_weights


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm= True, spec_norm= False, resample= None):
        super(ResidualBlock, self).__init__()
                
        # (Normalization) - Activation - Convolution -(Normalization) - Activation - Convolution
        self.batch_norm = batch_norm

        self.conv1, self.conv2, self.shortcut = self._make_conv_layer(in_channels, out_channels, kernel_size, stride, padding, batch_norm, spec_norm, resample)
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        '''
        if batch_norm:
            layers.append(CondBatchNorm2d(in_channels))
        layers.append(nn.ReLU(inplace= True))
        layers.append(conv1)
        if batch_norm:
            layers.append(CondBatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace= True))
        layers.append(conv2)

        self.features = nn.Sequential( *layers )
        '''
        self.nonlinearF = nn.ReLU(inplace= True)
        if batch_norm:
            self.cbn1 = CondBatchNorm2d(in_channels, 10, momentum=1.0)
            self.cbn2 = CondBatchNorm2d(out_channels, 10, momentum=1.0)
        
    def forward(self, x, labels):
        if self.shortcut == None:
            out = self.nonlinearF(x)
            out = self.conv1(out)
            out = self.nonlinearF(out)
            out = self.conv2(out)
            return out + x
        else:
            if self.batch_norm:
                out = self.cbn1(x, labels)
                out = self.nonlinearF(out)
                out = self.conv1(out)
                out = self.cbn2(out, labels)
                out = self.nonlinearF(out)
                out = self.conv2(out)
                return out + self.shortcut(x)
            else:
                out = self.nonlinearF(x)
                out = self.conv1(out)
                out = self.nonlinearF(out)
                out = self.conv2(out)
                return out + self.shortcut(x)
    
    
    # If you use batch norm, that means you don't need to use bias parameters in convolution layers
    def _make_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm, spec_norm, resample):
        
        if resample == 'up':
            
            conv1 = nn.Sequential(
                                    nn.UpsamplingNearest2d(scale_factor= 2),
                                    nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                                            
                                    )
            conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
            
                
            kernel_size, padding, bias = 1, 0, False
            shortcut = nn.Sequential(
                                        nn.UpsamplingNearest2d(scale_factor= 2),
                                        nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias= bias),
                                                
                                        )
                
                
        elif resample == 'down':

            if spec_norm:
                
                conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
                conv2 = nn.Sequential( 
                                        spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)),
                                        nn.AvgPool2d((2, 2))
                                        
                                        )
                
                kernel_size, padding, bias = 1, 0, False
                shortcut = nn.Sequential( 
                                            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias= bias)),
                                            nn.AvgPool2d((2, 2))
                                            
                                            )
                
                
            else:
                
                conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias= not batch_norm)
                conv2 = nn.Sequential( 
                                        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                                        nn.AvgPool2d((2, 2))
                                        
                                        )
                
                kernel_size, padding, bias = 1, 0, False
                shortcut = nn.Sequential( 
                                            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias= bias),
                                            nn.AvgPool2d((2, 2))
                                            
                                            )   
            
                
        elif resample == None:
            
            if spec_norm:
                
                conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
                conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
                kernel_size, padding, bias = 1, 0, False
                shortcut = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=bias))
            
            else:
            
                conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
                kernel_size, padding, bias = 1, 0, False
                shortcut = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=bias)

        return conv1, conv2, shortcut
    




class CriticFirstBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, spec_norm= True):
        super(CriticFirstBlock, self).__init__()

        layers = []
        
        conv1, conv2, self.shortcut = self._make_conv_layer(in_channels, out_channels, kernel_size, stride, padding, spec_norm)

        conv1.apply(init_weights)
        conv2.apply(init_weights)

        layers.append(conv1)
        layers.append(nn.ReLU(inplace= True))
        layers.append(conv2)


        self.features = nn.Sequential(*layers)



    def forward(self, x):
        return self.features(x) + self.shortcut(x)
        
        
    def _make_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, spec_norm):
        
        
        if spec_norm:
            
            conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            conv2 = nn.Sequential( 
                                    spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)),
                                    nn.AvgPool2d((2, 2))
                                                    
                                    )
            kernel_size, padding = 1, 0
            shortcut = nn.Sequential(
                                    nn.AvgPool2d((2, 2)),
                                    spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, 0))

                                    )
            
        else:
            
            conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            conv2 = nn.Sequential( 
                                    nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                                    nn.AvgPool2d((2, 2))
                                                    
                                    )
            kernel_size, padding = 1, 0
            shortcut = nn.Sequential(
                                    nn.AvgPool2d((2, 2)),
                                    nn.Conv2d(in_channels, out_channels, 1, stride, 0)
                                                    
                                    )   

        return conv1, conv2, shortcut


class Generator(nn.Module):
    
    def __init__(self, latent_vector_dim, base_dim, dataset_name, hr_multiplier= 2, num_residual= 3):
        super(Generator, self).__init__()
        
        self.dataset_name = dataset_name
        if self.dataset_name == 'CIFAR10':
            self.num_residual = 3
        elif self.dataset_name == 'STL10':
            self.num_residual = 3
        elif self.dataset_name == 'Place365':
            self.num_residual = 4
        
        self.hr_multiplier = hr_multiplier
        from math import log2
        self.log2_hr = int(log2(self.hr_multiplier))

        # Linear layer for sampled latent vector z
        if self.dataset_name == 'CIFAR10' or self.dataset_name == 'Place365':
            self.first = nn.Linear(in_features= latent_vector_dim, out_features= base_dim*(2**(num_residual)) * 4 * 4,)    # input    [NC] = [N x 128],
                                                                                                       # output   [NC] = [N x BASE_DIM * 4 * 4],
                                                                                                       # reshape[NCHW] = [N x BASE_DIM x 4 x 4],
        elif self.dataset_name == 'STL10':
            self.first = nn.Linear(in_features= latent_vector_dim, out_features= base_dim*(2**(num_residual)) * 6 * 6,)    # input    [NC] = [N x 128],
                                                                                                       # output   [NC] = [N x BASE_DIM * 6 * 6],
                                                                                                       # reshape[NCHW] = [N x BASE_DIM x 6 x 6],

        torch.nn.init.xavier_uniform_(self.first.weight, gain=2.**0.5)
        
        # Make residual blocks
        for i in range(self.num_residual):
            self.add_module('residual_' + str(i + 1), ResidualBlock(
                                                                    in_channels= base_dim*(2**(num_residual-i)),              # input [NCHW] = [N x BASE_DIM x 4 x 4],
                                                                    out_channels= base_dim*(2**(num_residual-i-1)),             # output[NCHW] = [N x BASE_DIM x 8 x 8],
                                                                    kernel_size= 3,
                                                                    stride= 1,                          # input [NCHW] = [N x BASE_DIM x 8 x 8],
                                                                    padding= 1,                         # output[NCHW] = [N x BASE_DIM x 16 x 16],
                                                                    resample= 'up'
                                                                    
                                                                    ))                                  # input [NCHW] = [N x BASE_DIM x 16 x 16],
                                                                                                        # output[NCHW] = [N x BASE_DIM x 32 x 32],
        # Make high resolution imgs
        # Upsampling
        # Batchnorm On/Off Switch will be needed
        
        for i in range(self.log2_hr):

            self.add_module('residual_hr_' + str(i + 1), ResidualBlock(
                                                                            in_channels= base_dim,      # input [NCHW] = [N x BASE_DIM x 32 x 32],
                                                                            out_channels= base_dim,     # output[NCHW] = [N x BASE_DIM x IMG_HR * 32 x IMG_HR * 32],
                                                                            kernel_size= 3,
                                                                            stride= 1,
                                                                            padding= 1,
                                                                            resample= 'up'
                                                                        ))
                                            
        self.output_hr = nn.Sequential(
                            
                            #nn.BatchNorm2d(base_dim, affine=False, momentum=1.),
                            nn.ReLU(inplace= True),                                                     
                            nn.Conv2d(    in_channels= base_dim, out_channels= 3,                       # input [NCHW] = [N x BASE_DIM x 128 x 128],
                                          kernel_size= 3, stride= 1, padding= 1),                       # output[NCHW] = [N x 3 x 128 x 128],
                            nn.Tanh(),
                            
                            )
        self.output_hr.apply(init_weights)

        # Make low resolution imgs
        # Downsampling -- Using average pooling
        self.downsample = nn.AvgPool2d(kernel_size= (self.hr_multiplier, self.hr_multiplier))           # input [NCHW] = [N x 3 xIMG_HR * 32 x IMG_HR * 32],
                                                                                                        # output[NCHW] = [N x 3 x 32 x 32],
        
        
        
    
    def forward(self, x, labels):
        
        if self.dataset_name == 'CIFAR10':
            out_gen = self.first(x).view(x.size(0), -1, 4, 4)
        elif self.dataset_name == 'STL10':
            out_gen = self.first(x).view(x.size(0), -1, 6, 6)
        elif self.dataset_name == 'Places365':
            out_gen = self.first(x).view(x.size(0), -1, 4, 4)
        
        for i in range(self.num_residual):
            layer_name = 'residual_' + str(i + 1)
            
            for name, module in self.named_children():
                if layer_name == name:
                    out_gen = module(out_gen, labels)
                    
        for i in range(self.log2_hr):
            layer_name = 'residual_hr_' + str(i + 1)
            
            for name, module in self.named_children():
                if layer_name == name:
                    out_gen = module(out_gen, labels)

        out_hr = self.output_hr(out_gen)
        out_lr = self.downsample(out_hr)
        
        return out_lr, out_hr
        
    

class Critic(nn.Module):
    
    def __init__(self, base_dim, num_cls= 10, num_residual= 2):
        super(Critic, self).__init__()
        
        self.num_residual = num_residual
        
        
        # First residual block

        self.conv_first = CriticFirstBlock(
                                                in_channels= 3,                                         # input [NCHW] = [N x 3 x 32 x 32],
                                                out_channels= base_dim,                                 # output[NCHW] = [N x BASE_DIM x 16 x 16],
                                                kernel_size= 3,
                                                stride= 1,
                                                padding= 1,
                                                spec_norm= True,
                                                )

        
        # Make residual block
        for i in range(self.num_residual):
            
            if i == 0:
                resample = 'down'
            else:
                resample = None
            
            self.add_module('residual_' + str(i + 1), 
                                                    ResidualBlock(
                                                                    in_channels= base_dim*(2**i),              # input [NCHW] = [N x BASE_DIM x 16 x 16],
                                                                    out_channels= base_dim*(2**(i+1)),             # output[NCHW] = [N x BASE_DIM x 8 x 8],
                                                                    kernel_size= 3,
                                                                    stride= 1,                          # input [NCHW] = [N x BASE_DIM x 8 x 8],
                                                                    padding= 1,                         # output[NCHW] = [N x BASE_DIM x 8 x 8],
                                                                    batch_norm= False,
                                                                    spec_norm= True,                    # input [NCHW] = [N x BASE_DIM x 8 x 8],
                                                                    resample= resample,                 # output[NCHW] = [N x BASE_DIM x 8 x 8],
                                                                    ))                                  
            
                                                                                                        
                                                                                                        

            self.fc = spectral_norm(nn.Linear(base_dim*2**num_residual, 1, bias=False))
            torch.nn.init.xavier_uniform_(self.fc.weight)
            self.fc_labels = spectral_norm(nn.Linear(num_cls, base_dim*2**num_residual, bias=False))
            torch.nn.init.normal_(self.fc_labels.weight, 0., 0.01)
            
            
        
    
    def forward(self, x, labels):
        
        out_critic = self.conv_first(x)
        
        for i in range(self.num_residual):
            layer_name = 'residual_' + str(i + 1)

            for name, module in self.named_children():
                if layer_name == name:
                    out_critic = module(out_critic, labels)
                                                                                                        # input [NCHW] = [N x BASE_DIM x 8 x 8],
        out_features = torch.sum(F.relu(out_critic, inplace= True), dim= (2, 3))                        # output[NCHW] = [N x BASE_DIM],
        
        out = self.fc(out_features)                                                                     # input [NCHW] = [N x BASE_DIM],
                                                                                                        # output[NCHW] = [N x NUM_CLASS],
        out_labels = self.fc_labels(labels)
        out_labels = out_labels * out_features
        out_labels = torch.sum(out_labels, dim=1)

        return out.squeeze() + out_labels.squeeze()
    
    
