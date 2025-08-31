import torch
from torch import nn 

import matplotlib

import matplotlib.pyplot as plt



# Step 1: Create example data
X = list(range(50))              # Features: 0 to 49
y = [i * 2 for i in X]           # Labels: each value is 2x of X

# Step 2: Train/test split
train_split = int(0.8 * len(X))  # 80% of data
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Step 3: Confirm the split sizes
print("Train/Test Sizes:")
print("X_train:", len(X_train), "y_train:", len(y_train))
print("X_test:", len(X_test), "y_test:", len(y_test))

# Step 4: Define a function to plot data and predictions
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data, and optional predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=40, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=40, label="Testing data")

    # Plot predictions if available
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=40, label="Predictions")

    # Show legend
    plt.legend(prop={"size": 14})
    plt.xlabel("X values")
    plt.ylabel("y values")
    plt.title("Train/Test Split Visualization")
    plt.grid(True)
    plt.show()

# Step 5: Call the function to visualize (no predictions yet)
plot_predictions()



