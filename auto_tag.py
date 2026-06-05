"""
Auto-tag unfilled fashion image metadata using CLIP zero-shot classification.
Designed for CPU-only / 16 GB RAM systems using ViT-B/32.
"""

import os
import json
import torch
import clip
import pandas as pd
from PIL import Image
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------

IMAGES_DIR = "dataset/images"
METADATA_PATH = "metadata.csv"
TAXONOMY_PATH = "taxonomy.json"
SAVE_EVERY = 25  # save progress every N images

TAG_FIELDS = [
    "category",
    "silhouette",
    "sleeve",
    "neckline",
    "color",
    "style",
    "pattern",
]

CSV_COLUMNS = [
    "image_id",
    "file_name",
    "category",
    "silhouette",
    "sleeve",
    "neckline",
    "color",
    "style",
    "pattern",
    "extra_notes",
]

# -------------------------
# CLIP PROMPT TEMPLATES
# -------------------------
# Good prompt engineering is crucial for CLIP zero-shot accuracy.
# Each field gets a template that gives CLIP semantic context.

PROMPT_TEMPLATES = {
    "category": "a photo of a {}",
    "silhouette": "a garment with a {} silhouette",
    "sleeve": "a garment with {}",
    "neckline": "a garment with a {} neckline",
    "color": "a {} colored garment",
    "style": "a {} style outfit",
    "pattern": "a garment with a {} pattern",
}

# Special handling: for some values, we can craft better prompts
PROMPT_OVERRIDES = {
    ("sleeve", "none"): "a garment with no sleeves visible, such as pants or a skirt",
    ("sleeve", "sleeveless"): "a sleeveless garment or tank top",
    ("neckline", "none"): "a garment with no neckline visible, such as pants or a skirt",
    ("neckline", "collared"): "a garment with a collar",
    ("category", "t-shirt"): "a photo of a t-shirt or tee",
    ("silhouette", "none"): "a garment with no distinct silhouette",
    ("color", "none"): "a garment with no distinct color",
    ("style", "none"): "a garment with no distinct style",
    ("pattern", "none"): "a garment with no distinct pattern",
    ("pattern", "plain"): "a plain solid color garment with no pattern",
    ("pattern", "textured"): "a garment with a textured fabric surface",
    ("silhouette", "fit and flare"): "a garment with a fitted top and flared bottom",
    ("silhouette", "regular fit"): "a regular fit garment, not too tight or loose",
    ("category", "top"): "a photo of a women's top or blouse",
}


def build_prompt(field: str, value: str) -> str:
    """Build a CLIP text prompt for a given field-value pair."""
    key = (field, value)
    if key in PROMPT_OVERRIDES:
        return PROMPT_OVERRIDES[key]
    template = PROMPT_TEMPLATES.get(field, "a photo of a {}")
    return template.format(value)


# -------------------------
# LOAD RESOURCES
# -------------------------

print("Loading taxonomy...")
with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
    taxonomy = json.load(f)

print("Loading metadata...")
if os.path.exists(METADATA_PATH):
    metadata_df = pd.read_csv(METADATA_PATH)
    for col in CSV_COLUMNS:
        if col not in metadata_df.columns:
            metadata_df[col] = ""
else:
    metadata_df = pd.DataFrame(columns=CSV_COLUMNS)

print("Loading CLIP ViT-B/32 (CPU)...")
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()


# -------------------------
# PRE-ENCODE TAXONOMY TEXT
# -------------------------

print("Encoding taxonomy text prompts...")
text_features_cache = {}

for field in TAG_FIELDS:
    options = taxonomy[field]
    prompts = [build_prompt(field, opt) for opt in options]
    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_feats = model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    text_features_cache[field] = {
        "options": options,
        "features": text_feats,
    }


# -------------------------
# HELPERS
# -------------------------

def is_field_filled(value) -> bool:
    """Check if a metadata field has a meaningful value."""
    if pd.isna(value):
        return False
    val = str(value).strip().lower()
    return val not in ["", "nan", "none"]


def classify_image(image_features: torch.Tensor, field: str) -> str:
    """Zero-shot classify an image for a given taxonomy field."""
    cache = text_features_cache[field]
    similarities = (image_features @ cache["features"].T).squeeze(0)
    best_idx = similarities.argmax().item()
    return cache["options"][best_idx]


def get_image_files():
    """Get list of all image files in the dataset."""
    return sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])


# -------------------------
# MAIN PROCESSING
# -------------------------

def main():
    global metadata_df

    image_files = get_image_files()
    print(f"\nFound {len(image_files)} images in dataset.")

    # Build a lookup for existing metadata
    existing_lookup = {}
    for _, row in metadata_df.iterrows():
        existing_lookup[row["file_name"]] = row

    # Count how many images need work
    needs_work = []
    for img_file in image_files:
        row = existing_lookup.get(img_file)
        if row is None:
            needs_work.append(img_file)
            continue
        # Check if any field is unfilled
        for field in TAG_FIELDS:
            if not is_field_filled(row.get(field, "")):
                needs_work.append(img_file)
                break

    print(f"Images already fully tagged: {len(image_files) - len(needs_work)}")
    print(f"Images needing auto-tagging: {len(needs_work)}")

    if not needs_work:
        print("\nAll images are already tagged! Nothing to do.")
        return

    print(f"\nProcessing {len(needs_work)} images...\n")

    tagged_count = 0

    for i, img_file in enumerate(tqdm(needs_work, desc="Auto-tagging")):

        # Load and encode image
        img_path = os.path.join(IMAGES_DIR, img_file)
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        except Exception as e:
            tqdm.write(f"  Skipping {img_file}: {e}")
            continue

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Get existing row or create new
        existing_row = existing_lookup.get(img_file)

        row_data = {}
        if existing_row is not None:
            row_data = dict(existing_row)
        else:
            row_data["file_name"] = img_file
            row_data["image_id"] = os.path.splitext(img_file)[0]
            row_data["extra_notes"] = ""

        # Fill only unfilled fields
        for field in TAG_FIELDS:
            current_val = row_data.get(field, "")
            if not is_field_filled(current_val):
                predicted = classify_image(image_features, field)
                row_data[field] = predicted

        # Update dataframe
        existing_index = metadata_df[
            metadata_df["file_name"] == img_file
        ].index

        if len(existing_index) > 0:
            for key, value in row_data.items():
                if key in metadata_df.columns:
                    metadata_df.loc[existing_index[0], key] = value
        else:
            metadata_df = pd.concat(
                [metadata_df, pd.DataFrame([row_data])],
                ignore_index=True,
            )
            # Update lookup for future reference
            existing_lookup[img_file] = pd.Series(row_data)

        tagged_count += 1

        # Periodic save
        if tagged_count % SAVE_EVERY == 0:
            metadata_df.to_csv(METADATA_PATH, index=False)
            tqdm.write(f"  [Checkpoint] Saved progress at {tagged_count}/{len(needs_work)}")

    # Final save
    metadata_df.to_csv(METADATA_PATH, index=False)
    print(f"\n[Done] Auto-tagged {tagged_count} images.")
    print(f"   Results saved to {METADATA_PATH}")
    print(f"   Use metadata_reviewer.py to review and correct predictions.")


if __name__ == "__main__":
    main()
