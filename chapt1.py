

#pytorch end to end workflow 

#what_were_covering = {1:"data (prepare the load)",
                     # 2:"build a model",
                      #3:"fitting the model to data(traiining)",
                      #4:"making predictions and evaluting a model (inference)",
                      #5:"saving and loading the model",
                      #6:"putting it all together"}



import torch 
from torch import nn   #nn contains all of pytorches building blocks for neural network
import matplotlib.pyplot as plt

## ( data preparing and loading )
#The data could be anything such as -excel spreadsheets,images,videos,audios,dna,texts

#machine learning is a game of two parts 
#1-to get data into a numerical representation 
#2-build a model to learn patters in that representation 


                                    




#using a linear regression formula to make a straight line 

#creating known parameters 
weight = 0.7
bias = 0.3

#create 
start = 0
end = 1
step = 0.02
x = torch.arange(start,  end,  step ).unsqueeze(dim=1) #unsqueeze adds an extra dimension here 
y= weight* x + bias 

print(x[:10], y[:10])


