from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
import random
import math

g_sparsity_slice = 32.0

def update_mask(mask, weights, layer_type):
    if layer_type != 'conv':
        return mask

    abs_weights = torch.abs(weights)
    g_sparsity_int = int(g_sparsity_slice)

    out_depth, in_depth, filter_width, filter_height = weights.shape
    max_weight = torch.max(abs_weights)
    sparsity_slice = g_sparsity_slice
    depth_iterations = int(math.ceil(out_depth/sparsity_slice))
    
#    print(g_sparsity_slice, g_sparsity_int)

    for f_w in range(filter_width):
        for f_h in range(filter_height):
            for i_d in range(in_depth):
                for d_i in range(depth_iterations):
                    start_index = d_i*g_sparsity_int
                    min_value = max_weight
                    min_index = -1

                    for i in range(start_index, min(start_index + g_sparsity_int, out_depth)):
                        if abs_weights[i, i_d, f_w, f_h] < min_value and mask[i, i_d, f_w, f_h] != 0:
                            min_value = abs_weights[i,i_d, f_w, f_h]
                            min_index = i

                    if min_index == -1:
                        print("Error ")
                    mask[min_index, i_d, f_w, f_h] = 0
                    #print(min_index)

    return mask

class NewMaskedLayer(nn.Linear):
    #initialize the Binary Layer where weights are binarized
    def __init__(self, input_dim, output_dim):
       super(NewMaskedLayer, self).__init__(input_dim, output_dim)
#       self.mask = torch.ones(self.weight.data.shape)
 
    def forward(self, x, update_flag):
        
        '''
        if update_flag == True:
            self.mask = update_mask(self.mask, self.weight.data, "linear")

        self.weight.data = self.mask * self.weight.data
        '''
        # compute layer operation
        out = super(NewMaskedLayer, self).forward(x)

        return out



class NewMaskedConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(NewMaskedConv2D, self).__init__(in_channels, out_channels, kernel_size, stride=stride)
        self.mask = torch.ones(self.weight.data.shape).cuda()
#        self.mask = torch.ones(self.weight.data.shape)


    def forward(self, x, update_flag):

        
        if update_flag == True:
            print("pruning weights")
            self.mask = update_mask(self.mask, self.weight.data, "conv")

        self.weight.data = self.mask * self.weight.data
        

        # compute layer operation
        out = super(NewMaskedConv2D, self).forward(x)


        return out
