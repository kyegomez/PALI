import torch
from pali.model import Pali

# # Initialize Pali model
# pali = Pali()

# Example 1: Caption an Image
# # Load images
# images = [torch.randn(1, 3, 256, 256) for _ in range(3)]

# for i, img in enumerate(images):
#     # Generate a caption for the image
#     prompt = torch.randint(0, 256, (1, 1024))
#     prompt_mask = torch.ones(1, 1024).bool()
#     output_text = torch.randint(0, 256, (1, 1024))

#     result = pali.process(img, prompt, output_text, prompt_mask)
#     print(f"Caption for image {i+1}: ", result)


# # # Example 2: Generate text based on another piece of text
# # Define prompt texts
# prompt_texts = ["Once upon a time", "In a galaxy far, far away", "It was a dark and stormy night"]

# for i, prompt_text in enumerate(prompt_texts):
#     prompt = torch.tensor([ord(c) for c in prompt_text]).unsqueeze(0)

#     # Generate text based on the prompt
#     output_text = torch.randint(0, 256, (1, 1024))

#     result = pali.process(None, prompt, output_text, None)
#     print(f"Generated text for prompt {i+1}: ", result)

pali = Pali()

prompt_text = "say hi to Kye"
model_name = "t5-small"  # specify the model name or path
generated_text = pali.generate(prompt_text, model_name=model_name)
print(generated_text)
