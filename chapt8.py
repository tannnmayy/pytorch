import torch 
import matplotlib.pyplot as plt
 
# 1. Construct a model that subclasses nn.Module
class CircleModelV0(nn.Module):

    def __init__():
        super().__init__()

        # 2. Create 2 nn.Linear layers capable of handling the shapes of our data
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features and up
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
