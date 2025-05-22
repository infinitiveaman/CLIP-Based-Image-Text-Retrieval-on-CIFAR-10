# 🔍 CLIP-Based Image-Text Retrieval on CIFAR-10 (200 Images)

This project implements a simple image-text retrieval engine using **OpenAI's CLIP model** on a **200-image subset** of the CIFAR-10 dataset. The system enables users to query images either by uploading an image or by providing a natural language text prompt.

---

## 🧩 Features

- **Two-Way Retrieval**:
  - **Text-to-Image**: Input a text description and retrieve the most relevant CIFAR-10 images.
  - **Image-to-Image**: Upload an image and find visually similar images from the dataset.

- **Efficient Search**:
  - All 200 CIFAR-10 images are preprocessed and encoded using CLIP.
  - Uses cosine similarity for fast nearest-neighbor search.

- **Interactive Interface**:
  - Built in **Google Colab** with visual outputs of top-k search results.

---

## 🧠 Model

- **CLIP (Contrastive Language–Image Pretraining)** by OpenAI
  - Encodes images and text into a shared embedding space.
  - Enables semantic similarity comparison across modalities.

---

## 📂 Dataset

- **CIFAR-10 (Subset)**
  - 200 images across 10 classes.
  - Each image is 32x32 pixels.

---


