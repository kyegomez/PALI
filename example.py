import torch
from pali.model import VitModel, Pali


#training data
img = torch.randn(1, 3, 256, 256)
print(img.shape)

prompt = torch.randint(0, 256, (1, 1024)) # prompt
print(prompt.shape)

prompt_mask = torch.ones(1, 1024).bool()
print(prompt_mask.shape)

output_text = torch.randint(0, 256, (1, 1024)) #target output text
print(output_text.shape)

#train
img_embeds = VitModel(
    img, 
    image_size=128,
    return_embeddings=True
)

loss = Pali(
    prompt,
    output_text,
    mask=prompt_mask,
    src_prepend_embeds=img_embeds # will prepend image embeddings
)

print(loss.backward())

