import torch
from pali import VitModel

# Random tensors
x = torch.randn(1, 3, 256, 256)

# Initialize model
model = VitModel()

# Forward pass
out = model(x)

# Print output shape
print(out.shape)
