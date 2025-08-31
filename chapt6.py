import torch 
from torch import nn   #nn contains all of pytorches building blocks for neural network
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
    


#building a training loop and a testing loop in pytorch 


'''

Thing Needed For Creating A Training Loop 

0-loop through the data 
1-forward pass (this invloves data moving through the models(forward))
2-calculate the loss (compare forward pass predictions to ground truth labels)
3-optimizer zero grad
4-loss backwards (move backwards through the network to calculate the gradients of eacj of the parameters of our modelw wrt to the loss)
5-optimizer step- use the optimizer to adjust our models parameters to try and improve the loss 
'''

#An epoch is one loop through the data...
 
#step-0
epochs =1


for epoch in range(epochs):

    model_0.train()

    model_0.eval() # turns off gradient tracking 



