[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Pali: A Multimodal Model
[![GitHub issues](https://img.shields.io/github/issues/kyegomez/pali)](https://github.com/kyegomez/pali/issues) 
[![GitHub forks](https://img.shields.io/github/forks/kyegomez/pali)](https://github.com/kyegomez/pali/network) 
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/pali)](https://github.com/kyegomez/pali/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/pali)](https://github.com/kyegomez/pali/blob/master/LICENSE)
[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/pali)](https://twitter.com/intent/tweet?text=Excited%20to%20introduce%20pali,%20the%20all-new%20robotics%20model%20with%20the%20potential%20to%20revolutionize%20automation.%20Join%20us%20on%20this%20journey%20towards%20a%20smarter%20future.%20%23RT1%20%23Robotics&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fpali)
[![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fpali)
[![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fpali&title=Introducing%20pali%2C%20the%20All-New%20Robotics%20Model&summary=pali%20is%20the%20next-generation%20robotics%20model%20that%20promises%20to%20transform%20industries%20with%20its%20intelligence%20and%20efficiency.%20Join%20us%20to%20be%20a%20part%20of%20this%20revolutionary%20journey%20%23RT1%20%23Robotics&source=)
![Discord](https://img.shields.io/discord/999382051935506503)
[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fpali&title=Exciting%20Times%20Ahead%20with%20pali%2C%20the%20All-New%20Robotics%20Model%20%23RT1%20%23Robotics) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fpali&t=Exciting%20Times%20Ahead%20with%20pali%2C%20the%20All-New%20Robotics%20Model%20%23RT1%20%23Robotics)
[![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Fpali&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=pali%2C%20the%20Revolutionary%20Robotics%20Model%20that%20will%20Change%20the%20Way%20We%20Work%20%23RT1%20%23Robotics)
[![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=I%20just%20discovered%20pali,%20the%20all-new%20robotics%20model%20that%20promises%20to%20revolutionize%20automation.%20Join%20me%20on%20this%20exciting%20journey%20towards%20a%20smarter%20future.%20%23RT1%20%23Robotics%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2Fpali)


The open source implementation of the Multi-Modality AI model from ["PaLI: Scaling Language-Image Learning in 100+ Languages"](https://arxiv.org/abs/2209.06794)

## 🌟 Appreciation
Big bear hugs 🐻💖 to *LucidRains* for the fab x_transformers and for championing the open source AI cause.

## 🚀 Quick Start

```bash
pip install pali-torch
```
---

## 🧙 Usage 
```python
import torch
from pali.model import VitModel, Pali

vit_module = VitModel()
pali_module = Pali()

img = torch.randn(1, 3, 256, 256)
prompt = torch.randint(0, 256, (1, 1024)) # prompt
prompt_mask = torch.ones(1, 1024).bool()
output_text = torch.randint(0, 256, (1, 1024)) #target output text

img_embeds = vit_module.process(img)
print(f"🎩 Image Magic: {img_embeds}")

loss = pali_module.process(prompt, output_text, prompt_mask, img_embeds)
loss = loss.backward()
print(f'🔮 Loss {loss}')
```
----

# Todo

- [ ] Make a table of datasets used in paper,
- [ ] Provide training script
- [ ] Provide usage/inference scripts


## 🎉 Features
- **Double the Power**: MT5 for text and ViT for images - Pali's the superhero we didn't know we needed! 💪📖🖼️
- **Winning Streak**: With roots in the tried-and-true MT5 & ViT, success is in Pali's DNA. 🏆
- **Ready, Set, Go**: No fuss, no muss! Get Pali rolling in no time. ⏱️
- **Easy-Peasy**: Leave the heavy lifting to Pali and enjoy your smooth sailing. 🛳️

## 🌆 Real-World Use-Cases

- **E-commerce**: Jazz up those recs! Understand products inside-out with images & descriptions. 🛍️
- **Social Media**: Be the smart reply guru for posts with pics & captions. 📱
- **Healthcare**: Boost diagnostics with insights from images & textual data. 🏥

----

## 📜 License
MIT

## 📚 Citation

```
@inproceedings{chen2022pali,
  title={PaLI: Scaling Language-Image Learning in 100+ Languages},
  author={Chen, Xi and Wang, Xiao},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```