from pali.transformer import ViTransformerWrapper, Encoder, UL2


class VitModel:
    """
    Vision Transformer Model.

    Args:
        image_size (int): The size of the input image (default: 256).
        patch_size (int): The size of each patch in the image (default: 32).
        dim (int): The dimension of the transformer model (default: 512).
        depth (int): The number of transformer layers (default: 6).
        heads (int): The number of attention heads in each transformer layer (default: 8).
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        image_size (int): The size of the input image.
        patch_size (int): The size of each patch in the image.
        dim (int): The dimension of the transformer model.
        depth (int): The number of transformer layers.
        heads (int): The number of attention heads in each transformer layer.
        vit (ViTransformerWrapper): The Vision Transformer model.

    """

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
        """
        Perform forward pass through the Vision Transformer model.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output embeddings from the Vision Transformer model.

        Raises:
            ValueError: If the input image is None or has an incorrect shape.

        """
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
        """
        Perform forward pass through the Vision Transformer model.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output embeddings from the Vision Transformer model.

        Raises:
            ValueError: If the input image is None or has an incorrect shape.

        """
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
    """
    Pali class represents the PALI model.

    Args:
        model_name (str): The name of the model (optional).
        image_size (int): The size of the input image (default: 256).
        patch_size (int): The size of each patch in the image (default: 32).
        dim (int): The dimensionality of the model (default: 512).
        depth (int): The depth of the model (default: 6).
        heads (int): The number of attention heads in the model (default: 8).
        enc_num_tokens (int): The number of tokens in the encoder (default: 256).
        enc_max_seq_len (int): The maximum sequence length for the encoder (default: 1024).
        dec_num_tokens (int): The number of tokens in the decoder (default: 256).
        dec_max_seq_len (int): The maximum sequence length for the decoder (default: 1024).
        enc_depth (int): The depth of the encoder (default: 6).
        enc_heads (int): The number of attention heads in the encoder (default: 8).
        dec_depth (int): The depth of the decoder (default: 6).
        dec_heads (int): The number of attention heads in the decoder (default: 8).
    """

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
        self.tokenizer = None
        self.vit_model = VitModel(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
        )

        self.ul = UL2(
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
        result = self.ul(prompt, output, mask=mask, src_prepend_embeds=img_embeds)

        return result
