
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_IMAGES_DIR = (
    PROJECT_ROOT / "dataset" / "images"
)

DEFAULT_METADATA_PATH = (
    PROJECT_ROOT / "metadata.csv"
)

DEFAULT_TAXONOMY_PATH = (
    PROJECT_ROOT / "taxonomy.json"
)

DEFAULT_REVIEW_OUTPUT = (
    PROJECT_ROOT / "metadata_ai_review.csv"
)

MODEL_NAME = "microsoft/Florence-2-base"


# -------------------------
# MODEL
# -------------------------

def load_model():
    """Load Florence model."""

    print("Loading Florence model...")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dtype = (
        torch.float16
        if device == "cuda"
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    print(
        f"Model loaded on {device}"
    )

    return (
        model,
        processor,
        device,
        dtype
    )


# -------------------------
# IMAGE CAPTION
# -------------------------

def generate_caption(
    image_path,
    model,
    processor,
    device,
    dtype
):
    """Generate fashion caption."""

    image = Image.open(
        image_path
    ).convert("RGB")

    prompt = (
        "<MORE_DETAILED_CAPTION>"
    )

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    processed_inputs = {}

    for key, value in inputs.items():

        if (
            value.dtype == torch.float
            and device == "cuda"
        ):
            processed_inputs[key] = (
                value.to(device)
                .to(dtype)
            )
        else:
            processed_inputs[key] = (
                value.to(device)
            )

    with torch.no_grad():

        generated_ids = model.generate(
            input_ids=processed_inputs[
                "input_ids"
            ],
            pixel_values=processed_inputs[
                "pixel_values"
            ],
            max_new_tokens=120,
            num_beams=3
        )

    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    parsed = (
        processor.post_process_generation(
            output,
            task="<MORE_DETAILED_CAPTION>",
            image_size=image.size
        )
    )

    caption = parsed.get(
        "<MORE_DETAILED_CAPTION>",
        ""
    ).lower()

    # -------------------------
    # CLEANUP
    # -------------------------

    remove_words = [
        "woman",
        "man",
        "girl",
        "boy",
        "standing",
        "posing",
        "wearing",
        "background",
        "white background",
        "person",
        "smiling",
        "young",
    ]

    for word in remove_words:
        caption = caption.replace(
            word,
            ""
        )

    caption = " ".join(
        caption.split()
    )

    return caption


# -------------------------
# TAXONOMY
# -------------------------

def load_taxonomy(
    taxonomy_path
):
    with open(
        taxonomy_path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# -------------------------
# RULE MATCHING
# -------------------------

def match_value(
    caption,
    allowed_values,
    synonym_map=None
):
    """Match caption to taxonomy."""

    if synonym_map is None:
        synonym_map = {}

    caption = caption.lower()

    # Exact taxonomy match
    for value in allowed_values:

        if (
            value.lower()
            == "none"
        ):
            continue

        if (
            value.lower()
            in caption
        ):
            return value

    # Synonym mapping
    for (
        keyword,
        mapped
    ) in synonym_map.items():

        if keyword in caption:

            if (
                mapped
                in allowed_values
            ):
                return mapped

    return "none"


def map_metadata(
    caption,
    taxonomy
):
    """Convert caption → metadata."""

    category_synonyms = {
        "overall": "jumpsuit",
        "overalls": "jumpsuit",
        "romper": "romper",
        "dress": "dress",
        "shirt": "shirt",
        "t shirt": "t-shirt",
        "tee": "t-shirt",
        "hoodie": "hoodie",
        "coat": "coat",
        "jacket": "jacket",
        "pants": "pants",
        "trousers": "pants",
        "jeans": "pants",
        "skirt": "skirt",
        "shorts": "shorts",
        "top": "top",
        "blouse": "top",
        "cami": "top",
        "tank top": "top",
        "off shoulder": "top",
        "off-the-shoulder": "top",
        "sweater": "sweater",
        "cardigan": "cardigan",
    }

    sleeve_synonyms = {
        "sleeveless": "sleeveless",
        "long sleeve": "long sleeve",
        "short sleeve": "short sleeve",
        "full sleeve": "long sleeve",
        "puff sleeve": "puff sleeve",
        "bell sleeve": "bell sleeve",
        "batwing": "batwing",
    }

    neckline_synonyms = {
        "v-neck": "v-neck",
        "v neck": "v-neck",
        "round neck": "round neck",
        "crew neck": "round neck",
        "square neck": "square",
        "halter": "halter",
        "collar": "collared",
        "high neck": "high neck",
        "sweetheart": "sweetheart",
        "strapless": "strapless",
        "off shoulder": "off-shoulder",
        "off-the-shoulder": "off-shoulder",
    }

    silhouette_synonyms = {
        "oversized": "oversized",
        "loose fit": "relaxed",
        "relaxed fit": "relaxed",
        "fitted": "fitted",
        "cropped": "cropped",
        "bodycon": "bodycon",
        "a-line": "a-line",
        "wide leg": "wide-leg",
        "straight": "straight",
    }

    pattern_synonyms = {
        "floral": "floral",
        "striped": "striped",
        "stripe": "striped",
        "checkered": "checkered",
        "checked": "checkered",
        "plaid": "checkered",
        "paisley": "paisley",
        "polka": "polka dot",
        "graphic": "graphic print",
        "embroidered": "embroidered",
        "lace": "lace",
        "corduroy": "solid",
    }

    style_synonyms = {
        "formal": "formal",
        "minimal": "minimal",
        "sporty": "sporty",
        "streetwear": "streetwear",
        "casual": "casual",
        "romantic": "romantic",
        "bohemian": "bohemian",
    }

    color_synonyms = {
        "black": "black",
        "white": "white",
        "brown": "brown",
        "blue": "blue",
        "green": "green",
        "pink": "pink",
        "red": "red",
        "yellow": "yellow",
        "grey": "grey",
        "gray": "grey",
        "beige": "beige",
        "cream": "cream",
    }

    metadata = {
        "category": match_value(
            caption,
            taxonomy["category"],
            category_synonyms
        ),
        "silhouette": match_value(
            caption,
            taxonomy["silhouette"],
            silhouette_synonyms
        ),
        "sleeve": match_value(
            caption,
            taxonomy["sleeve"],
            sleeve_synonyms
        ),
        "neckline": match_value(
            caption,
            taxonomy["neckline"],
            neckline_synonyms
        ),
        "color": match_value(
            caption,
            taxonomy["color"],
            color_synonyms
        ),
        "style": match_value(
            caption,
            taxonomy["style"],
            style_synonyms
        ),
        "pattern": match_value(
            caption,
            taxonomy["pattern"],
            pattern_synonyms
        ),
    }

    metadata[
        "extra_notes"
    ] = caption

    return metadata


# -------------------------
# MAIN PIPELINE
# -------------------------

def auto_metadata():
    """Generate AI review CSV."""

    taxonomy = (
        load_taxonomy(
            DEFAULT_TAXONOMY_PATH
        )
    )

    metadata_df = pd.read_csv(
        DEFAULT_METADATA_PATH
    )

    existing_files = set(
        metadata_df[
            "file_name"
        ]
        .astype(str)
        .str.strip()
    )

    image_paths = sorted(
        DEFAULT_IMAGES_DIR.glob("*")
    )

    new_images = [
        path
        for path in image_paths
        if path.name
        not in existing_files
    ]

    print(
        f"Found "
        f"{len(new_images)} "
        f"new images"
    )

    if not new_images:
        print(
            "No new images found."
        )
        return

    (
        model,
        processor,
        device,
        dtype
    ) = load_model()

    rows = []

    for (
        i,
        image_path
    ) in enumerate(
        new_images,
        start=1
    ):

        print(
            f"[{i}/"
            f"{len(new_images)}] "
            f"{image_path.name}"
        )

        try:

            caption = (
                generate_caption(
                    image_path,
                    model,
                    processor,
                    device,
                    dtype
                )
            )

            print(
                "Caption:",
                caption
            )

            metadata = (
                map_metadata(
                    caption,
                    taxonomy
                )
            )

            rows.append({
                "image_id":
                    image_path.stem,
                "file_name":
                    image_path.name,
                **metadata
            })

        except Exception as e:

            print(
                f"Failed: "
                f"{image_path.name}"
            )

            print(e)

    review_df = (
        pd.DataFrame(rows)
    )

    review_df = review_df[
        [
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
    ]

    review_df.to_csv(
        DEFAULT_REVIEW_OUTPUT,
        index=False
    )

    print(
        "\nSaved review file:"
    )

    print(
        DEFAULT_REVIEW_OUTPUT
    )