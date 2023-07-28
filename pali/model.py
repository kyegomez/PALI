import torch
from transformers import MT5Model, ViTModel

class PaLIModel(torch.nn.Module):
    def __init__(self, text_encoder_model='google/mt5-large', vision_encoder_model='google/vit-large-patch16-224'):
        super(PaLIModel, self).__init__()
        
        self.text_encoder_decoder = MT5Model.from_pretrained(text_encoder_model)
        self.vision_encoder = ViTModel.from_pretrained(vision_encoder_model)

    def forward(self, text_input, image_input):
        vision_output = self.vision_encoder(image_input)
        text_output = self.text_encoder_decoder(input_ids=text_input, encoder_outputs=vision_output.last_hidden_state)
        return text_output

model = PaLIModel()

