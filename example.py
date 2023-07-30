import torch
from pali.model import vit, pali


#training data
img = torch.randn(1, 3, 256, 256)
prompt = torch.randint(0, 256, (1, 1024)) # prompt
prompt_mask = torch.ones(1, 1024).bool()
output_text = torch.randint(0, 256, (1, 1024)) #target output text

#train
img_embeds = vit(
    img, 
    return_embeddings=True
)

loss = pali(
    prompt,
    output_text,
    mask=prompt_mask,
    src_prepend_embeds=img_embeds # will prepend image embeddings
)

loss.backward()

