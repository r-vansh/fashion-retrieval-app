import os
import torch
import clip
from PIL import Image
import pickle

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
# CREATE EMBEDDINGS
# -------------------------

image_paths = []
image_embeddings = []

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
        )

        image = preprocess(
            Image.open(path)
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            embedding = (
                model.encode_image(
                    image
                )
            )

        image_paths.append(
            path
        )

        image_embeddings.append(
            embedding.cpu()
        )

print(
    f"Processed "
    f"{len(image_paths)} images"
)

# -------------------------
# SAVE
# -------------------------

with open(
    "embeddings.pkl",
    "wb"
) as f:

    pickle.dump(
        (
            image_paths,
            image_embeddings
        ),
        f
    )

print(
    "Embeddings saved!"
)