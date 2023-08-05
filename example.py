import torch
from pali.model import VitModel, Pali

# usage
vit_module = VitModel()
pali_module = Pali()

#training data
img = torch.randn(1, 3, 256, 256)
prompt = torch.randint(0, 256, (1, 1024)) # prompt
prompt_mask = torch.ones(1, 1024).bool()
output_text = torch.randint(0, 256, (1, 1024)) #target output text

img_embeds = vit_module.process(img)

loss = pali_module.process(prompt, output_text, prompt_mask, img_embeds)

loss = loss.backward()
print(f'loss: {loss}')

