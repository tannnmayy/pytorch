import torch
from torch import nn 
import matplotlib.pyplot as plt


#splitting data into training and test sets 
#(one of the most important concepts there are )

# Create a train/test split




# Example data
X = list(range(50))         # Features: 0 to 49
y = [i * 2 for i in X]      # Labels: just 2x the feature value

# Create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
print(len(X_train), len(y_train), len(X_test), len(y_test))
