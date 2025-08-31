#  Fine-tuning VLMs via GRPO for Automatic Makeup Extraction
**📍 This is a project done during my internship at B*Factory | Jun 2025 - Aug 2025**<br>
😚 Special thanks to senior developers **Adrià Arrufat** and **Saad Imran** for their support and guidance throughout this project.

This project aims to automatically extract makeup looks from company photoshoots and allow customers to easily try them on their own faces using LViton’s virtual try-on system. <br>

The core idea is to fine-tune a Vision-Language Model (VLM) via GRPO (Generative Reinforcement Policy Optimization) to predict makeup attributes (lip, blush, eyeshadow colors) in a structured JSON format.<br>

Since a complete makeup look involves many parameters, the training was conducted progressively: starting with **lips only** (simpler) and later extending to **lips, blush, and eyeshadow.** <br><br>



## ⚙️ Experimental Settings
### Data
- Dataset size: ~10K
- Sources: Combination of Ameli makeup photoshoots + FFHQ faces
### Base Model
- Qwen/Qwen2.5-VL-7B-Instruct
### Reward
- Self defined. Detailed description below



## 💄 Lips Only
### Reward
### Results

## 🎨 Lips, Blush, Eyeshadow
### Reward
### Results











