import torch
from pali.model import ViTransformerWrapper, Encoder, XTransformer, AndromedaEmbedding
#pali composes of 
#1, vision transformer 
#2. Encoder decoder transformer(X transform)

vit = ViTransformerWrapper(
    image_size=256,
    patch_size=32,
    attn_layers = Encoder(
        dim=512,
        depth=6,
        heads=8
    )
)

pali = XTransformer(
    dim=512,
    enc_num_tokens=256,
    enc_depth=6,
    embedding_provider=AndromedaEmbedding(),
    enc_heads=8,
    enc_max_seq_len=1024,
    dec_num_tokens=256,
    dec_depth=6,
    dec_heads=8,
    dec_max_seq_len=1024
)

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

