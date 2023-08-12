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


🙌 Dive into the world of Pali, a rockstar model that's got the chops to groove to both text and image tunes! By blending the magic of MT5 📜 and ViT 🖼️, Pali is here to redefine multimodal data vibes.

## 🌟 Special Shoutout 
Big bear hugs 🐻💖 to *LucidRains* for the fab x_transformers and for championing the open source AI cause.

## 🚀 Quick Start

```bash
pip install pali-torch
```

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

## 🎉 Features
- **Double the Power**: MT5 for text and ViT for images - Pali's the superhero we didn't know we needed! 💪📖🖼️
- **Winning Streak**: With roots in the tried-and-true MT5 & ViT, success is in Pali's DNA. 🏆
- **Ready, Set, Go**: No fuss, no muss! Get Pali rolling in no time. ⏱️
- **Easy-Peasy**: Leave the heavy lifting to Pali and enjoy your smooth sailing. 🛳️

## 📐 Model Blueprint

Think of Pali as a swanky cocktail 🍹 - MT5 brings the text zest while ViT adds the image zing. Together, they craft a blend that’s pure magic! Whether it's MT5's adaptability or ViT's image smarts, Pali packs a punch. 🥊

## 🌆 Real-World Use-Cases

- **E-commerce**: Jazz up those recs! Understand products inside-out with images & descriptions. 🛍️
- **Social Media**: Be the smart reply guru for posts with pics & captions. 📱
- **Healthcare**: Boost diagnostics with insights from images & textual data. 🏥

# Contributing to Pali 🤖🌟

First off, big high fives 🙌 and thank you for considering a contribution to Pali! Your help and enthusiasm can truly elevate this project. Whether you're fixing bugs 🐛, adding features 🎁, or just providing feedback, every bit matters! Here's a step-by-step guide to make your contribution journey smooth:

## 1. Set the Stage 🎬

**Fork the Repository:** Before you dive in, create a fork of the Pali repository. This gives you your own workspace where you can make changes without affecting the main project.

1. Go to the top right corner of the Pali repo.
2. Click on the "Fork" button. 

Boom! You now have a copy on your GitHub account.

## 2. Clone & Set Up 🚀

**Clone Your Fork:** 
```bash
git clone https://github.com/YOUR_USERNAME/pali.git
cd pali
```

**Connect with the Main Repo:** To fetch updates from the main Pali repository, set it up as a remote:
```bash
git remote add upstream https://github.com/original_pali_repo/pali.git
```

## 3. Make Your Magic ✨

Create a new branch for your feature, bugfix, or whatever you're looking to contribute:
```bash
git checkout -b feature/my-awesome-feature
```

Now, dive into the code and sprinkle your magic!

## 4. Stay Updated 🔄

While you're working, the main Pali repository might have updates. Keep your local copy in sync:

```bash
git fetch upstream
git merge upstream/main
```

## 5. Share Your Brilliance 🎁

Once you've made your changes:

1. **Stage & Commit:**
   ```bash
   git add .
   git commit -m "Add my awesome feature"
   ```

2. **Push to Your Fork:**
   ```bash
   git push origin feature/my-awesome-feature
   ```

3. **Create a Pull Request:** Head back to your fork on GitHub, and you'll see a "New Pull Request" button. Click on it!

## 6. The Review Dance 💃🕺

Once your PR is submitted, our Pali team will review it. They might have questions or feedback. Stay engaged, discuss, and make any needed changes. Collaboration is key! 🤝

## 7. Celebrate 🎉

After review and any necessary tweaks, your contribution will be merged. Pat yourself on the back and celebrate! 🎊

## 8. Spread the Word 📢

Share about your contribution with your network. The more the merrier! Plus, it feels good to show off a bit, right? 😉

Remember, every contribution, no matter how small or large, is valued and appreciated. It's the collective effort that makes open-source so vibrant and impactful. Thanks for being a part of the Pali adventure! 🌟🚀

----

## 📜 License

Pali grooves under the MIT License. Dive into the [LICENSE](LICENSE) for all the deets.

## 💌 Drop Us A Line

Got Qs? 🤔 Ping us with an issue or visit our superstar [kyegomez](https://github.com/kyegomez) on GitHub.

## 📚 Citation

```
@inproceedings{chen2022pali,
  title={PaLI: Scaling Language-Image Learning in 100+ Languages},
  author={Chen, Xi and Wang, Xiao},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

Let's co-create, learn, and grow with Pali! 🌱🚀🎉