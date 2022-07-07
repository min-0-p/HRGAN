# -*- coding: utf-8 -*-
"""
Created on Thu May  6 22:40:03 2021

@author: MinYoung
"""

import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init

from torchvision.transforms import functional as TF


def gradient_penalty(critic, real, fake, labels, k= 1, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, labels)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - k) ** 2)
    return gradient_penalty


def get_mobilenet_score(mobilenet, x, img_size):

    # resize img size if img size < 224
    if img_size < 224:
        preprocess = TF.resize(x, 224)
    else: preprocess = x

    prediction = F.softmax(mobilenet(preprocess), dim=1)

    p_y = torch.mean(prediction, dim= 0)
    e = prediction * torch.log(prediction / p_y)
    kl = torch.mean(torch.sum(e, dim= 1), dim= 0)
    
    return torch.exp(kl)


class _NormBase_CBN(Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps',
                     'num_features', 'affine']
    num_features: int
    num_labels: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        num_features: int,
        num_labels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ) -> None:
        super(_NormBase_CBN, self).__init__()
        self.num_features = num_features
        self.num_labels = num_labels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Linear(self.num_labels, self.num_features, bias=False)
            torch.nn.init.normal_(self.weight.weight, 0., 0.02)
            self.bias = nn.Linear(self.num_labels, self.num_features, bias=False)
            torch.nn.init.normal_(self.bias.weight, 0., 0.001)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[operator]
            self.running_var.fill_(1)  # type: ignore[operator]
            self.num_batches_tracked.zero_()  # type: ignore[operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        '''
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)
        '''

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_NormBase_CBN, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class _BatchNorm_CBN(_NormBase_CBN):

    def __init__(self, num_features, num_labels, eps=1e-5, momentum=None, affine=True,
                 track_running_stats=True):
        super(_BatchNorm_CBN, self).__init__(
            num_features, num_labels, eps, momentum, affine, track_running_stats)
        self.dummy_w = torch.ones(self.num_features).cuda()
        self.dummy_b = torch.zeros(self.num_features).cuda()

    def forward(self, input: Tensor, labels: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.affine:
            self.weight_CBN = self.weight(labels) + 1.
            self.bias_CBN = self.bias(labels)

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        mean = input.mean([0, 2, 3])  # along channel axis
        var = input.var([0, 2, 3])

        #current_mean = mean.view([1, self.num_features, 1, 1]).expand_as(input)
        #current_var = var.view([1, self.num_features, 1, 1]).expand_as(input)
        current_weight = self.weight_CBN.view([-1, self.num_features, 1, 1]).expand_as(input)
        current_bias = self.bias_CBN.view([-1, self.num_features, 1, 1]).expand_as(input)
        
        

        return current_weight * F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.dummy_w, self.dummy_b, True, exponential_average_factor, self.eps) + current_bias



class CondBatchNorm2d(_BatchNorm_CBN):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))



def one_hot_embedding(labels, num_classes, device='cpu'):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    labels = labels.type(torch.long)
    y = torch.eye(num_classes).to(device)
    return y[labels]

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=2.**0.5)