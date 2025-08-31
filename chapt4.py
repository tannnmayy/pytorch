import torch 
from torch import nn 
import matplotlib
import matplotlib.pyplot as plt

##BUILDING A LINEAR REGRESSION MODEL

class LinearRegressionModel(nn.module): #almost everything in pytorch is inheritance from nn module
    def_init_self(self):
    self.weights = nn.Parameter(torch.randn(1,
                                            requires_grad=true,  #if the parameters requires gradient, hence 
                                            dtype=torch.float))
    

    self.bias = nn.Parameter(torch.randn(1,
                                         requires_grad=true,
                                         dtype=torch.float))
    #
    def forard(self, x: torch.tensor) -> torch.tensor:
        return self.weights * x + self.bias #(this is that linear regression formula written in my stupid ass book)
    
