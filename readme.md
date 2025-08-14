# 🖼️ Large language models based Image Restoration Agent

We explores the use of pretrained language models for **image restoration tasks** such as denoising, inpainting, and deblurring — all without retraining or modifying the underlying model weights.

---

## 🎯 Objective

Repurpose LLMs for pixel-level reasoning tasks by:

- Encoding image corruption descriptions into structured prompts
- Guiding restoration steps using autoregressive generation
- Outputting either image tokens or restoration decisions

ResLLM demonstrates that LLMs can **reason over visual artifacts** when paired with symbolic or token-based image representations.

---

## 🛠️ Key Features

- **Training-free**: No model fine-tuning required
- **Tokenized image input**: Converts images into structured sequences
- **Restoration via prompting**: Generates restoration steps as text or token predictions
- **Flexible decoding**: Outputs can be parsed into pixel values or restoration instructions

---

## 📦 Installation

```bash
cd resllm
pip install -r requirements.txt
```

---

## 🖼️ Example Usage

Run the restoration pipeline with a corrupted image:

```bash
python run_resllm.py --input noisy_image.png --task denoise
```

For visualization:

```bash
python visualize.py --input original.png --output restored.png
```

