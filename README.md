#  Fine-tuning VLMs via GRPO for Automatic Makeup Extraction
📍 This is a project done during my internship at B*Factory | Jun 2025 - Aug 2025

This repository contains the implementation and experiments for automatic makeup extraction using Vision-Language Models (VLMs).

The ultimate goal of this project is to enable customers to easily try on the exact makeup looks from a company’s photoshoots directly on their own faces.
By automatically extracting the color and style information from editorial makeup images and applying them to personal selfies via LViton, 
we aim to bridge the gap between inspiration and personalization.


## 🚀 Project Overview
### Goal
- Extract accurate makeup information from beauty photoshoots.
- Allow customers to instantly try on these makeup looks on their own faces with LViton’s virtual try-on system

### Approach
- Fine-tune a VLM using GRPO (Generative Reinforcement Policy Optimization)
- Continual training: <br>
    - **Phase 1** → Lips only (since the base model was relatively small and a full makeup look contains too many attributes to learn at once)
    - **Phase 2** → Lips + Blush + Eyeshadow
- Custom reward function balancing format correctness and color accuracy
