import torch
from torch import nn
import matplotlib.pyplot as plt



print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Create known parameters
weight = 0.8
bias = 0.6

# 2. Prepare the data
start, end, step = 0, 1, 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)  # shape: [n_samples, 1]
y = weight * X + bias

# 3. Build the model using nn.Module (correct capital `M`)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    
model_0 = LinearRegressionModel()

# 4. Define loss function and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# 5. Training and evaluation loop
epochs = 80  # You can increase this for better learning

#to track values hence we can compare fure experiments and past experiments  
epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):
    model_0.train()  # Put model into training mode

    # Forward pass
    y_pred = model_0(X)
    loss = loss_fn(y_pred, y)
    print(f"loss:{loss}")
 
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Eval step
    model_0.eval()
    with torch.inference_mode(): 
        y_pred_eval = model_0(X)
        loss_eval = loss_fn(y_pred_eval, y)

    # Log progress every 10 epochs
    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss_values)
        test_loss_values.append(test_loss_values)

        print(f"Epoch {epoch:3d}: Train Loss = {loss.item():.4f} | Eval Loss = {loss_eval.item():.4f}")

# 6. Plot the results
with torch.inference_mode():
    y_pred_line = model_0(X)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="True data")
plt.plot(X, y_pred_line, color='red', label="Model predictions")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Panshul Arora BKL ")
plt.show()
 
torch.save
torch.load
torch.nn.Module.load_state_dict

#saving a model in pytorch 
#just fucking save it madarchod 



