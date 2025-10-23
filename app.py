# app.py
# -----------------------------------------
# Multilingual CLIP Image Retrieval Demo (CPU Optimized)
# -----------------------------------------

import torch
import clip
from sentence_transformers import SentenceTransformer
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import gradio as gr
from PIL import Image

# -----------------------------------------
# Force CPU
# -----------------------------------------
device = "cpu"

# -----------------------------------------
# Load models
# -----------------------------------------
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()  # eval mode
text_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1', device=device)

# -----------------------------------------
# Load CIFAR-10 dataset (first 200 test images)
# -----------------------------------------
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466,0.4578275,0.40821073),
                         (0.26862954,0.26130258,0.27577711))
])

dataset = CIFAR10(root="./data", train=False, download=True)
images = [transform(dataset[i][0]) for i in range(200)]
original_images = [dataset[i][0] for i in range(200)]
labels = [dataset[i][1] for i in range(200)]
image_batch = torch.stack(images).to(device)

# Precompute CLIP image embeddings
with torch.no_grad():
    image_embeddings = clip_model.encode_image(image_batch).float()
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

# -----------------------------------------
# Image-to-Image search with similarity
# -----------------------------------------
def image_to_image_with_scores(uploaded_image, topk=5):
    uploaded_image = uploaded_image.convert("RGB").resize((224,224))
    image_tensor = preprocess(uploaded_image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = clip_model.encode_image(image_tensor).float()
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
    
    similarities = (query_embedding @ image_embeddings.T).squeeze(0)
    top_indices = similarities.topk(topk).indices.cpu().numpy()
    
    results = []
    for idx in top_indices:
        sim_score = similarities[idx].item()
        results.append((original_images[idx], f"{class_names[labels[idx]]}\nScore: {sim_score:.2f}"))
    return results

# -----------------------------------------
# Text-to-Image search (multilingual) with similarity
# -----------------------------------------
def text_to_image_with_scores(query, topk=5):
    text_embedding = text_model.encode([query], convert_to_tensor=True, device=device).clone()
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    
    similarities = (text_embedding @ image_embeddings.T).squeeze(0)
    top_indices = similarities.topk(topk).indices.cpu().numpy()
    
    results = []
    for idx in top_indices:
        sim_score = similarities[idx].item()
        results.append((original_images[idx], f"{class_names[labels[idx]]}\nScore: {sim_score:.2f}"))
    return results

# -----------------------------------------
# Gradio Interface
# -----------------------------------------
with gr.Blocks(title="Multilingual CLIP Image Search (CPU)") as demo:
    gr.Markdown("##  CLIP + M-CLIP Image Retrieval Demo\nUpload an image or type text in any language.")
    
    with gr.Tab(" Image-to-Image Search"):
        image_input = gr.Image(label="Upload Image", type="pil")
        image_gallery = gr.Gallery(label="Results", show_label=False, columns=5, height=200)
        image_button = gr.Button("Find Similar ")
        image_button.click(image_to_image_with_scores, inputs=image_input, outputs=image_gallery)
    
    with gr.Tab("🈴 Text-to-Image Search"):
        text_input = gr.Textbox(label="Enter query (any language, e.g., कुत्ता, horse, chien, 狗)")
        text_gallery = gr.Gallery(label="Results", show_label=False, columns=5, height=200)
        text_button = gr.Button("Search ")
        text_button.click(text_to_image_with_scores, inputs=text_input, outputs=text_gallery)

demo.launch()
