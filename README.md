#  Fine-tuning VLMs via GRPO for Automatic Makeup Extraction
🌎 **This is a project done during my internship at [B*Factory](https://www.linkedin.com/company/bfactory-ai/posts/?feedView=all) | Jun 2025 - Aug 2025**<br>
📍 Special thanks to senior developers **Adrià Arrufat** and **Saad Imran** for their support and guidance throughout this project.

This project aims to automatically extract makeup looks from company photoshoots and allow customers to easily try them on their own faces using LViton’s virtual try-on system. <br>

The core idea is to fine-tune a Vision-Language Model (VLM) via [GRPO (Generative Reinforcement Policy Optimization)](https://arxiv.org/abs/2402.03300) to predict makeup attributes (lip, blush, eyeshadow colors) in a structured JSON format.  e.g.: 
```
[
  {"shape": "LIP_FULL_BASIC", "color": "#DC3813"},
  {"shape": "BLUSHER_CENTER_WIDE_BASIC", "color": "#F07749"},
  {"shape": "EYESHADOW_OVEREYE_FULL_BASIC", "color": "#511515"}
]
```
<br>

Since a complete makeup look involves many parameters, the training was conducted progressively: starting with **lips only** (simpler) and later extending to **lips, blush, and eyeshadow.** <br><br>



## ⚙️ Experimental Settings
### Data
- Dataset size: 10K
- Paired dataset of company's internal makeup parameters  + [FFHQ](https://github.com/NVlabs/ffhq-dataset)
### Base Model
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
### Reward
- Self defined. Detailed description below <br><br>


## 💄 Lips Only
### Reward
### Results
<img src='./assets/lips_0.png' width=580><br>
<img src='./assets/lips_1.png' width=580><br><br>
<img src='./assets/lips_2.png' width=580><br>
<img src='./assets/lips_3.png' width=580><br><br>

## 🎨 Lips, Blush, Eyeshadow
### Reward
### Results
<img src='./assets/full_0.png' width=580><br><br>
<img src='./assets/full_1.png' width=580>










