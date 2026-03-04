# Reading List

Papers I want to read, am reading, or have finished.

Use `/reading-list add <arxiv_id>` to add papers here.

<!-- Format: each paper is a ## section with metadata -->

---

## Accelerating Scientific Research with Gemini: Case Studies and Common Techniques
- **ArXiv**: 2602.03837 | https://arxiv.org/abs/2602.03837
- **Authors**: David P. Woodruff, Vincent Cohen-Addad, Lalit Jain et al. (36 authors, Google DeepMind)
- **Status**: 📖 Unread
- **Categories**: cs.CL, cs.AI
- **Conference**: Not yet published
- **Added**: 2026-03-05
- **Notes**: 151页案例集，记录 Gemini Deep Think 与理论CS/经济/物理研究者的真实协作。三大模式：①对抗评审（SNARGs密码学漏洞检测：迭代自我纠正5步提示法）②跨域迁移（Max-Cut→测度论/Stone-Weierstrass；Steiner树→Kirszbraun延伸定理）③神经符号循环（宇宙弦谱积分：~600分支树搜索+Python自动验证，80%自动剪枝，找到O(1)闭合解）。关键技巧：负向提示、上下文去识别、脚手架推理。

---

## Geometric-Mean Policy Optimization
- **ArXiv**: 2507.20673 | https://arxiv.org/abs/2507.20673
- **Authors**: Yuzhong Zhao, Yue Liu, Junpeng Liu, Furu Wei et al. (12 authors, Microsoft Research affiliates)
- **Status**: 📖 Unread
- **Categories**: cs.LG, cs.CL
- **Conference**: 🤗 HF Daily Pick
- **Added**: 2026-02-25
- **Notes**: GMPO — 用几何均值替代 GRPO 的算术均值，解决 token 重要性采样比极端值导致的训练不稳定。GMPO-7B 在数学推理 benchmark 上平均超过 GRPO +4.1%，多模态推理 +1.4%。即插即用，直接替换 GRPO。代码: https://github.com/callsys/GMPO

---

## Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models
- **ArXiv**: 2507.08128 | https://arxiv.org/abs/2507.08128
- **Authors**: Arushi Goel, Sreyan Ghosh, Jaehyeon Kim et al. (NVIDIA + UMD)
- **Status**: 📖 Unread
- **Categories**: cs.SD, cs.CL, cs.LG
- **Conference**: NeurIPS 2025 Spotlight
- **Added**: 2026-02-25
- **Notes**: AF3 — 统一处理语音/音效/音乐的开源大音频语言模型，AF-Whisper 编码器 + Qwen2.5-7B + Streaming TTS，50M 音频文本对训练，在 20+ benchmark SOTA，支持最长 10 分钟音频推理。模型: nvidia/audio-flamingo-3

---

## SmolDocling: An Ultra-Compact Vision-Language Model for End-to-End Multi-Modal Document Conversion
- **ArXiv**: 2503.11576 | https://arxiv.org/abs/2503.11576
- **Authors**: IBM Research + Hugging Face
- **Status**: 📖 Unread
- **Categories**: cs.CV, cs.CL
- **Conference**: ICCV 2025
- **Added**: 2026-02-25
- **Notes**: 仅 256M 参数端到端文档转换 VLM，引入 DocTags 格式，性能匹敌 27 倍大的模型。0.35秒/页，0.489GB VRAM，支持 Transformers/VLLM/ONNX/MLX。代码/表格/方程 F1 均超 Qwen2.5-VL(7B)。模型: ds4sd/SmolDocling-256M-preview

---

## Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance
- **ArXiv**: 2510.24711 | https://arxiv.org/abs/2510.24711
- **Authors**: ali-vilab team (Alibaba DAMO Academy) et al.
- **Status**: 📖 Unread
- **Categories**: cs.CV, cs.LG
- **Conference**: ICLR 2026
- **Added**: 2026-02-25
- **Notes**: ProMoE — 两步路由(Conditional + Prototypical)解决视觉MoE专家分配问题，ImageNet FID 2.79 vs 密集baseline 3.56。官方代码: https://github.com/ali-vilab/ProMoE

---
