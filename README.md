# PaLI: Scaling Language-Image Learning in 100+ Languages

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Model Architecture](#model-architecture)
- [Training and Data](#training-and-data)
- [Results](#results)
- [Fairness, Bias, and Other Considerations](#fairness-bias-and-other-considerations)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Introduction
PaLI is a unified language-image model trained to perform many tasks and in over 100 languages. It is designed to maximize the value by increasing the perceived likelihood of success and dream outcome while minimizing the time to success and the effort required. The model addresses a wide range of tasks in the language-image, language-only, and image-only domain using the same API.

## Installation
```bash
git clone https://github.com/kyegomez/PaLI.git
cd PaLI
pip install -r requirements.txt
```

## How to Use
```python
from pali import PaLI
model = PaLI.load_pretrained("path_to_model")
output = model.predict(image="path_to_image", text="input_text")
```

## Model Architecture
The PaLI model architecture is simple, reusable, and scalable. It consists of a Transformer encoder for processing the input text, and an auto-regressive Transformer decoder that generates the output text. The input to the Transformer encoder also includes "visual words" that represent an image processed by a Vision Transformer (ViT). PaLI uses weights from previously-trained uni-modal vision and language models, such as mT5-XXL and large ViTs. 

## Training and Data
To train PaLI, we constructed WebLI, a multilingual language-image dataset built from images and text available on the public web. The data collection process scaled the WebLI dataset to 10 billion images and 12 billion alt-texts in 109 languages. The model is trained with a mixture of pre-training tasks using the JAX with Flax using the open-sourced T5X and Flaxformer framework.

## Results
PaLI outperforms the state-of-the-art approaches (including SimVLM, CoCa, GIT2, Flamingo, BEiT3) on multiple vision-and-language tasks. It shows improved performance across visual-, language-, and vision-language tasks.

## Fairness, Bias, and Other Considerations
We strive to avoid creating or reinforcing unfair bias within large language and image models. The paper includes a data card and model card to be transparent about the data used and how the model used those data. Results of demographic analyses of the dataset are also included.

## Contributing
Contributions to this project are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License
This project is licensed under the terms of the Apache 2.0 license. See [LICENSE](LICENSE) for more details.

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