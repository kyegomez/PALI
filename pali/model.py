from pali.transformer import ViTransformerWrapper, Encoder, XTransformer

class VitModel:
    def __init__(self, 
                 image_size=256, 
                 patch_size=32, 
                 dim=512, 
                 depth=6, 
                 heads=8, 
                 *args, **kwargs):
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

        self.depth = depth
        self.heads = heads
        self.vit = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                heads=heads
            )
        )
    
        
    def process(self, img):
        if img is None:
            raise ValueError('Input image cannot be None')
        if img.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError('Input image must have the shape [*, 3, {}, {}]'.format(self.image_size, self.image_size))
        
        return self.vit(img, return_embeddings=True)
    
class Transformer:
    def __init__(
            self, 
            dim=512, 
            enc_num_tokens=256, 
            enc_depth=6, 
            enc_heads=8, 
            enc_max_seq_len=1024, 
            dec_num_tokens=256, 
            dec_depth=6, 
            dec_heads=8, 
            dec_max_seq_len=1024
        ):
        super().__init__()
        self.pali = XTransformer(
            dim=dim,
            enc_num_tokens=enc_num_tokens,
            enc_depth=enc_depth,
            enc_heads=enc_heads,
            enc_max_seq_len=enc_max_seq_len,
            dec_num_tokens=dec_num_tokens,
            dec_depth=dec_depth,
            dec_heads=dec_heads,
            dec_max_seq_len=dec_max_seq_len
        )

    def process(
            self, 
            prompt, 
            output_text, 
            mask, 
            img_embeds
        ):
        if prompt is None or output_text is None or mask is None or img_embeds is None:
            raise ValueError('None of the input parameters can be None')

        return self.pali(
            prompt,
            output_text,
            mask=mask,
            src_prepend_embeds=img_embeds
        )
    



class Pali:
    def __init__(
        self,
        image_size=256,
        patch_size=32,
        dim=512,
        depth=6,
        heads=8,
        enc_num_tokens=256,
        enc_max_seq_len=1024,
        dec_num_tokens=256,
        dec_max_seq_len=1024
    ):
        self.vit_model = VitModel(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads
        )

        self.pali_model = Transformer(
            dim=dim,
            enc_num_tokens=enc_num_tokens,
            enc_depth=depth,
            enc_heads=heads,
            enc_max_seq_len=enc_max_seq_len,
            dec_num_tokens=dec_num_tokens,
            dec_depth=depth,
            dec_heads=heads,
            dec_max_seq_len=dec_max_seq_len
        )

    def process(
        self,
        img,
        prompt,
        output,
        mask
    ):
        img_embeds = self.vit_model.process(img)
        result = self.pali_model.process(prompt, output, mask, img_embeds)
        return result
    
    def generate(
        self,
        text,
        seq_len=1024,
        mask=None,
        attn_Mask=None
    ):
        self.pali_model.generate()





# # usage
# vit_module = ViTModule()
# pali_module = PaliModule()

# #training data
# img = torch.randn(1, 3, 256, 256)
# prompt = torch.randint(0, 256, (1, 1024)) # prompt
# prompt_mask = torch.ones(1, 1024).bool()
# output_text = torch.randint(0, 256, (1, 1024)) #target output text

# img_embeds = vit_module.process(img)

# loss = pali_module.process(prompt, output_text, prompt_mask, img_embeds)

# loss.backward()
