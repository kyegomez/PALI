import torch
from pali import Pali

model = Pali()

img = torch.randn(1, 3, 256, 256)  # Image tensor
prompt = torch.randint(0, 256, (1, 1024))  # Text integer tensor
output_text = torch.randint(0, 256, (1, 1024))  # Target Text integer tensor

out = model.forward(img, prompt, output_text, mask=None)


print(out)
