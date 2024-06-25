import math
import logging
import torch
import torch.nn.functional as F
from torch import nn


logger = logging.getLogger(__file__)

class ICNN(nn.Module):
    def __init__(self, layer_sizes, activation=F.relu_):
        super().__init__()
        self.W = nn.ParameterList([nn.Parameter(torch.Tensor(l, layer_sizes[0])) 
                                   for l in layer_sizes[1:]])
        self.U = nn.ParameterList([nn.Parameter(torch.Tensor(layer_sizes[i+1], layer_sizes[i]))
                                   for i in range(1,len(layer_sizes)-1)])
        self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(l)) for l in layer_sizes[1:]])
        self.act = activation
        self.reset_parameters()
        logger.info(f"Initialized ICNN with {self.act} activation")
        self.rehu = ReHU(1.0)


    def reset_parameters(self):
        # copying from PyTorch Linear
        for W in self.W:
            nn.init.kaiming_uniform_(W, a=5**0.5)
        for U in self.U:
            nn.init.kaiming_uniform_(U, a=5**0.5)
        for i,b in enumerate(self.bias):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W[i])
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        z = F.linear(x, self.W[0], self.bias[0])
        z = self.act(z)

        for W,b,U in zip(self.W[1:-1], self.bias[1:-1], self.U[:-1]):
            z = F.linear(x, W, b) + F.linear(z, F.softplus(U)) / U.shape[0]
            z = self.act(z)
        
        V_res = F.linear(x, self.W[-1], self.bias[-1]) + F.linear(z, F.softplus(self.U[-1])) / self.U[-1].shape[0]

        z_zero = F.linear(torch.zeros_like(x), self.W[0], self.bias[0])
        z_zero = self.act(z_zero)

        for W,b,U in zip(self.W[1:-1], self.bias[1:-1], self.U[:-1]):
            z_zero = F.linear(torch.zeros_like(x), W, b) + F.linear(z_zero, F.softplus(U)) / U.shape[0]
            z_zero = self.act(z_zero)
        V_zero = F.linear(torch.zeros_like(x), self.W[-1], self.bias[-1]) + F.linear(z_zero, F.softplus(self.U[-1])) / self.U[-1].shape[0]

        return self.rehu(V_res - V_zero)
    
class ReHU(nn.Module):
    """ Rectified Huber unit"""
    def __init__(self, d):
        super().__init__()
        self.a = 1/d
        self.b = -d/2

    def forward(self, x):
        return torch.max(torch.clamp(torch.sign(x)*self.a/2*x**2,min=0,max=-self.b),x+self.b)