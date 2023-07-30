# Pali: A Multimodal Model

## Introduction

Pali is a powerful, state-of-the-art multimodal model designed to process and generate responses from both text and image inputs. By combining the text processing prowess of MT5 and image processing capabilities of ViT (Vision Transformer), Pali offers an enhanced understanding of multimodal data. This model was developed by [kyegomez](https://github.com/kyegomez) and this README provides a comprehensive guide on how to install, understand, and use it.

## Value Proposition

Pali delivers a transformative approach to multimodal understanding, opening up new possibilities for applications and research. 

- **Maximized Outcome**: By leveraging two leading models (MT5 for text, ViT for images), Pali offers superior multimodal understanding, potentially leading to enhanced performance in numerous applications.
- **High Perceived Likelihood of Success**: Built upon tested and proven architectures (MT5 and ViT), the likelihood of achieving successful results with Pali is high.
- **Minimized Time to Success**: With an easy installation process and clear usage instructions, you can start working with Pali quickly.
- **Minimal Effort & Sacrifice**: Pali handles the complexities of multimodal data, simplifying your data processing pipeline. 

## Installation

Pali can be installed via pip:

```bash
pip install pali
```

Additionally, the dependencies can be manually installed:

```bash
pip install torch transformers
```

# Usage
Here's an example of how to use the Pali model:

```python
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

```

## Model Architecture

Pali combines the MT5 and ViT models to create a comprehensive, multimodal model. 

MT5 (a text-to-text transformer model) is used for processing text. The power of MT5 lies in its ability to handle different tasks with a simple change in the task's prefix, making it incredibly flexible.

ViT (Vision Transformer), on the other hand, is used for processing images. ViT treats image patches as sequences of data, similar to text, allowing the model to apply transformers to image processing.

In Pali, the output of the ViT model is used as the encoder outputs for the MT5 model. This combination allows Pali to leverage the strengths of both models, offering a powerful approach to multimodal understanding.

## Commercial Use Cases

Pali's unique design makes it highly suitable for a variety of commercial applications:

- **E-commerce**: Pali can be used to improve product recommendation systems by understanding both product images and descriptions.
- **Social Media**: Understanding and generating responses for posts containing both text and images.
- **Healthcare**: Analyzing medical images and associated text data for improved diagnostics.

## Contributing

Contributions to Pali are welcome! Please feel free to open an issue or pull request on the GitHub repository.

## License

Pali is provided under the MIT License. See the LICENSE file for details.

## Contact

If you have any questions or issues, please open a GitHub issue or reach out to [kyegomez](https://github.com/kyegomez).

## Citation
If you use this code or use our pre-trained model, please cite the PaLI paper:

```
@inproceedings{chen2022pali,
  title={PaLI: Scaling Language-Image Learning in 100+ Languages},
  author={Chen, Xi and Wang, Xiao},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```