import os
import torch
import clip
from PIL import Image
import pickle
import cv2
import numpy as np

# -------------------------
# PATHS
# -------------------------

images_folder = "dataset/images"

# -------------------------
# LOAD CLIP
# -------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model, preprocess = clip.load(
    "ViT-B/32",
    device=device
)

# -------------------------
# EDGE DETECTION HELPER
# -------------------------

def extract_edges(pil_img):
    # Convert PIL Image to Grayscale numpy array
    img_np = np.array(pil_img)
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Smooth to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Invert binary image to get black lines on white background
    edges_inverted = cv2.bitwise_not(edges)
    
    # Convert back to PIL Image and ensure RGB format for CLIP preprocess
    return Image.fromarray(edges_inverted).convert("RGB")

# -------------------------
# CREATE EMBEDDINGS
# -------------------------

image_paths = []
image_embeddings = []
edge_embeddings = []

files = sorted(
    os.listdir(images_folder)
)

for file in files:

    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        path = os.path.join(
            images_folder,
            file
        ).replace(
            "\\",
            "/"
        )

        pil_img = Image.open(path)
        
        # Standard embedding
        image = preprocess(
            pil_img
        ).unsqueeze(0).to(device)

        # Edge-extracted embedding
        edge_pil = extract_edges(pil_img)
        edge_image = preprocess(
            edge_pil
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            embedding = (
                model.encode_image(
                    image
                )
            )
            embedding /= (
                embedding.norm(
                    dim=-1,
                    keepdim=True
                )
            )

            edge_emb = (
                model.encode_image(
                    edge_image
                )
            )
            edge_emb /= (
                edge_emb.norm(
                    dim=-1,
                    keepdim=True
                )
            )

        image_paths.append(
            path
        )

        image_embeddings.append(
            embedding.cpu()
        )
        
        edge_embeddings.append(
            edge_emb.cpu()
        )

print(
    f"Processed "
    f"{len(image_paths)} images"
)

# -------------------------
# SAVE
# -------------------------

data_to_save = {
    "image_paths": image_paths,
    "image_embeddings": image_embeddings,
    "edge_embeddings": edge_embeddings
}

with open(
    "embeddings.pkl",
    "wb"
) as f:

    pickle.dump(
        data_to_save,
        f
    )

print(
    "Embeddings saved!"
)