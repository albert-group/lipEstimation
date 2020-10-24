""" Starting point of the script is a saving of all singular values and vectors
in custom_save/

We perform the 100-optimization implemented in optim_nn_pca_greedy
"""
import math
import torch
import torch.nn as nn
import torchvision

import numpy as np

from lipschitz_utils import *
from max_eigenvalue import k_generic_power_method

#from experiments.bruteforce_optim import optim_nn_pca_greedy
from seqlip import optim_nn_pca_greedy

#import models.mnist_5_custom as custom_network # verification
import models.custom_network as custom_network

# network
net = custom_network.net()
input_size = custom_network.input_size
convs = [0, 3, 6, 8, 10] # indices of conv
lins = [1, 4, 6] # indices of linear layers
#net = net.cuda()

n_sv = 200 # number of singular values to use in the k_generic_power_method in the max_eigenvalue function

for p in net.parameters():
    p.requires_grad = False

compute_module_input_sizes(net, input_size)

# indices of convolutions and linear layers
layers = []
for i, function in enumerate(net.functions):
    if isinstance(function, nn.Conv2d):
        layers.append('Conv2d-' + str(i))
    elif isinstance(function, nn.Linear):
        layers.append('Linear-' + str(i))

lip_spectral = 1
lip = 1

##########################
# convolutions and linears
##########################
for i in range(len(layers) - 1):
    print('Dealing with ', layers[i])
    U = torch.load('custom_save/feat-left-sing-' + layers[i])
    U = torch.cat(U[:n_sv], dim=0).view(n_sv, -1)
    su = torch.load('custom_save/feat-singular-' + layers[i])
    su = su[:n_sv]

    V = torch.load('custom_save/feat-right-sing-' + layers[i+1])
    V = torch.cat(V[:n_sv], dim=0).view(n_sv, -1)
    sv = torch.load('custom_save/feat-singular-' + layers[i+1])
    sv = sv[:n_sv]
    print('Ratio layer i  : {:.4f}'.format(float(su[0] / su[-1])))
    print('Ratio layer i+1: {:.4f}'.format(float(sv[0] / sv[-1])))

    U, V = U.cpu(), V.cpu()

    # first layer
    if i == 0:
        sigmau = torch.diag(torch.Tensor(su))
    else:
        sigmau = torch.diag(torch.sqrt(torch.Tensor(su)))

    # last layer in iteration
    # Does this assume the last layer doesn't have a ReLU?
    if i == len(convs) - 2:
        sigmav = torch.diag(torch.Tensor(sv))
    else:
        sigmav = torch.diag(torch.sqrt(torch.Tensor(sv)))

    expected = sigmau[0,0] * sigmav[0,0]
    print('Expected: {}'.format(expected))
    lip_spectral *= float(expected)

    try:
        curr, _ = optim_nn_pca_greedy(sigmav @ V, U.t() @ sigmau)
        print('Approximation: {}'.format(curr))
        lip *= float(curr)
    except:
        print('Probably something went wrong...')
        lip *= float(expected)


print('Lipschitz spectral: {}'.format(lip_spectral))
print('Lipschitz approximation: {}'.format(lip))
