import torch.nn as nn
from config import *
from utils import device

class deepNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(deepNetwork, self).__init__()
        # 3 layer network
        self.fc = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.Sigmoid(),
            nn.Linear(64,128),
            nn.Sigmoid(),
            nn.Linear(128,128),
            nn.Sigmoid(),
            nn.Linear(128,128),
            nn.Sigmoid(),
            nn.Linear(128,output_dim),
        )
    def forward(self, x):
        return self.fc(x)

class kernelEncoder(nn.Module):
    def __init__(self, N, input_dim, output_dim):
        super(kernelEncoder, self).__init__()
        self.net = deepNetwork(input_dim, output_dim)
    def forward(self, x):
        return self.net(x)

class kernelDncoder(nn.Module):
    def __init__(self, N, input_dim, output_dim):
        super(kernelDncoder, self).__init__()
        self.net = deepNetwork(input_dim, output_dim)
    def forward(self, x):
        return self.net(x)

# global
encoder = kernelEncoder(N, x_dim, embedding_dim).double().to(device)
decoder = kernelDncoder(N, embedding_dim, x_dim).double().to(device)