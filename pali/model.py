import torch
from transformers import MT5Model, ViTModel, AutoTokenizer

class PaliTokenizer:
    """
    A tokenizer class for the Pali model.
    
    Attributes:
        tokenizer (AutoTokenizer): The tokenizer for text.
    """
    def __init__(self, text_encoder_model='google/mt5-large'):
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_model)

    def tokenize_texts(self, texts):
        """
        Tokenize given texts.
        
        Args:
            texts (str): The text to be tokenized.
        
        Returns:
            The tokenized texts.
        """
        return self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids

    def tokenize(self, sample):
        """
        Tokenizes given sample.
        
        Args:
            sample: The sample to be tokenized.
        
        Returns:
            A dictionary containing the tokenized text.
        """
        return {
            "text_tokens": self.tokenize_texts(sample["text"]),
        }


class Pali(torch.nn.Module):
    """
    The Pali model, a multimodal model combining an MT5 model for text and a ViT model for images.

    Attributes:
        text_encoder_decoder (MT5Model): The text encoder-decoder model.
        vision_encoder (ViTModel): The vision encoder model.
    """
    def __init__(self, text_encoder_model='google/mt5-large', vision_encoder_model='google/vit-large-patch16-224'):
        super(Pali, self).__init__()

        self.text_encoder_decoder = MT5Model.from_pretrained(text_encoder_model)
        self.vision_encoder = ViTModel.from_pretrained(vision_encoder_model)

    def forward(self, text_input, image_input):
        """
        Forward pass through the model.
        
        Args:
            text_input (torch.Tensor): The text input to the model.
            image_input (torch.Tensor): The image input to the model.
        
        Returns:
            The output from the text encoder-decoder model.
        """
        vision_output = self.vision_encoder(image_input)
        text_output = self.text_encoder_decoder(input_ids=text_input, encoder_outputs=vision_output.last_hidden_state)
        return text_output
