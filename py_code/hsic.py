"""ReBias
Copyright (c) 2020-present NAVER Corp.
MIT license

Python Implementation of the finite sample estimator of Hilbert-Schmidt Independence Criterion (HSIC)
We provide both biased estimator and unbiased estimators (unbiased estimator is used in the paper)

implement 1
https://github.com/clovaai/rebias/blob/master/criterions/hsic.py

"""
from py_code.utils import device
from py_code.config import *
from py_code.utils import to_cpu

import torch
import torch.nn as nn
import numpy as np

from scipy.linalg import fractional_matrix_power


def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x):
    '''
        input torch.Float
        return Gaussian kernel matrix
    '''
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_)

class HSIC(nn.Module):
    """Base class for the finite sample estimator of Hilbert-Schmidt Independence Criterion (HSIC)
    ..math:: HSIC (X, Y) := || C_{x, y} ||^2_{HS}, where HSIC (X, Y) = 0 iif X and Y are independent.

    Empirically, we use the finite sample estimator of HSIC (with m observations) by,
    (1) biased estimator (HSIC_0)
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        :math: (m - 1)^2 tr KHLH.
        where K_{ij} = kernel_x (x_i, x_j), L_{ij} = kernel_y (y_i, y_j), H = 1 - m^{-1} 1 1 (Hence, K, L, H are m by m matrices).
    """
    def __init__(self, N):
        super(HSIC, self).__init__()
        self.N = N
        H = torch.eye(N,dtype=torch.float) - torch.ones((N,N), dtype = torch.double) / N
        self.H = H.double().to(device)
        self.D = torch.ones((N,N), dtype = torch.double)

        self.estimator = self.biased_estimator
        
    def _kernel_x(self, X):
        # Gaussian Kernel
        return GaussianKernelMatrix(X)

    def _kernel_y(self, Y):
        # linear kernel
        return Y @ Y.T

    def biased_estimator(self, input1, input2, fix_D = True):
        """Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """

        K = self._kernel_x(input1)

        if not fix_D:
            self.update_D(K)

        # ~kx = (D^-0.5) * K * (D^-0.5)
        K = self.D @ K @ self.D 
        
        Y = self._kernel_y(input2)

        # H = I - (1/n)*1*1
        KH = K @ self.H
        YH = Y @ self.H
        return torch.trace((KH @ YH) / (self.N - 1) ** 2)

    def update_D(self, K):
        '''
            given K(x) update D
        '''
        # D = diag(x*1_N)
        D = np.diag(to_cpu(K).mean(1))
        # D ^ -0.5
        self.D = torch.tensor(fractional_matrix_power(np.linalg.inv(D), 0.5),device=device)

    def selfTest(self):
        '''
            self test by random value
        '''
        N = self.N
        x = torch.rand(N,2).double().to(device)
        # expected not independent => value high
        y = x
        print('dependent(high): ',self.forward(x, y,False).item())

        # expected independent => 0
        y = torch.rand(N,2).double().to(device)
        print('independent(low): ', self.forward(x,y).item())

    def forward(self, input1, input2, fix_D = True, **kwargs):
        ''' class entry point '''
        return self.estimator(input1, input2, fix_D)

hsic = HSIC(BATCH_SIZE)