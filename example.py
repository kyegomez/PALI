import torch
from pali.model import VitModel, Pali


# training data

img = torch.randn(1, 3, 256, 256)               # images
prompt = torch.randint(0, 256, (1, 1024))       # prompt
prompt_mask = torch.ones(1, 1024).bool()        # prompt text mask
output_text = torch.randint(0, 256, (1, 1024))  # target output text

#train
img_embeds = VitModel(
    img, 
    return_embeddings=True
)

loss = Pali(
    prompt,
    output_text,
    mask=prompt_mask,
    src_prepend_embeds=img_embeds # will prepend image embeddings
)

print(loss.backward())

