Perfect! I’ve updated your GitHub-ready README with your live demo link. You can copy-paste this directly into your repo.

---

# 🔍 Multilingual CLIP-Based Image-Text Retrieval

Interactive **image-text retrieval system** using **OpenAI CLIP** and **Multilingual CLIP (M-CLIP)**.
Users can search images **by uploading an image** or providing **text in any language**, with results showing **top-K matches and similarity scores**.

---

## 🧩 Features

* **Two-Way Retrieval**

  * **Image-to-Image:** Find visually similar images by uploading a sample.
  * **Text-to-Image (Multilingual):** Enter queries in English, Hindi, Spanish, Chinese, etc., to retrieve matching images.

* **Similarity Scores:** Each result shows a cosine similarity score for transparency.

* **Interactive Interface:** Built with **Gradio** for easy browser-based interaction.

* **Fast Retrieval:** Precomputes embeddings for all images for efficient nearest-neighbor search.

---

## 🧠 Models

* **CLIP (OpenAI):** Encodes images and text into a shared embedding space.
* **Multilingual CLIP (M-CLIP):** Enables cross-lingual text-to-image search.

---

## 📂 Dataset

* **CIFAR-10 Subset:** 200 images across 10 classes (`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`).
* Each image resized to 224x224 for CLIP processing.
* Can be extended to **custom datasets**.

---

## ⚡ Deployment

* **Live Demo:** Hosted on **Hugging Face Spaces** with Gradio.
* Works on **CPU**, no GPU required for free hosting.

---

## 🛠️ Requirements

```text
torch
torchvision
ftfy
regex
tqdm
gradio
sentence-transformers
Pillow
matplotlib
numpy
git+https://github.com/openai/CLIP.git
```

---

## 🚀 Usage

1. **Image-to-Image Search**

   * Upload an image → get top-K visually similar images with similarity scores.

2. **Text-to-Image Search (Multilingual)**

   * Enter a text query in any language → get top-K matching images with similarity scores.

---

## 🌐 Live Demo

[View Demo on Hugging Face Spaces](https://huggingface.co/spaces/aman-iitp/mclip-search)

---

