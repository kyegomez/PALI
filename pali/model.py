from pali.transformer import ViTransformerWrapper, Encoder, UL2


class VitModel:
    def __init__(
        self, image_size=256, patch_size=32, dim=512, depth=6, heads=8, *args, **kwargs
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

        self.depth = depth
        self.heads = heads
        self.vit = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(dim=dim, depth=depth, heads=heads),
        )

    def __call__(self, img):
        if img is None:
            raise ValueError("Input image cannot be None")
        if img.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(
                "Input image must have the shape [*, 3, {}, {}]".format(
                    self.image_size, self.image_size
                )
            )

        return self.vit(img, return_embeddings=True)

    def forward(self, img):
        if img is None:
            raise ValueError("Input image cannot be None")
        if img.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(
                "Input image must have the shape [*, 3, {}, {}]".format(
                    self.image_size, self.image_size
                )
            )

        return self.vit(img, return_embeddings=True)


class Pali:
    def __init__(
        self,
        model_name=None,
        image_size=256,
        patch_size=32,
        dim=512,
        depth=6,
        heads=8,
        enc_num_tokens=256,
        enc_max_seq_len=1024,
        dec_num_tokens=256,
        dec_max_seq_len=1024,
        enc_depth=6,
        enc_heads=8,
        dec_depth=6,
        dec_heads=8,
    ):
        self.model_name = model_name
        self.tokenizer = None
        self.vit_model = VitModel(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
        )

        self.ul2 = UL2(
            dim=dim,
            enc_num_tokens=enc_num_tokens,
            enc_depth=enc_depth,
            enc_heads=enc_heads,
            enc_max_seq_len=enc_max_seq_len,
            dec_num_tokens=dec_num_tokens,
            dec_depth=dec_depth,
            dec_heads=dec_heads,
            dec_max_seq_len=dec_max_seq_len,
        )

    def forward(self, img, prompt, output, mask):
        """Get the image embeddings"""
        img_embeds = self.vit_model.forward(img)

        """Get the output text embeddings"""
        result = self.ul2(prompt, output, mask=mask, src_prepend_embeds=img_embeds)

        return result
