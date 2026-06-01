import os
import shutil
import random
import pandas as pd
from pathlib import Path


# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DEEPFASHION_ROOT = os.path.join(
    PROJECT_ROOT,
    "DeepFashion"
)

IMG_ROOT = os.path.join(
    DEEPFASHION_ROOT,
    "img"
)

BENCHMARK_ROOT = os.path.join(
    DEEPFASHION_ROOT,
    "Category and Attribute Prediction Benchmark"
)

ANNO_COARSE = os.path.join(
    BENCHMARK_ROOT,
    "Anno_coarse"
)

CATEGORY_IMG_FILE = os.path.join(
    ANNO_COARSE,
    "list_category_img.txt"
)

CATEGORY_CLOTH_FILE = os.path.join(
    ANNO_COARSE,
    "list_category_cloth.txt"
)

# =====================================================
# SAFE OUTPUTS (REVIEW ONLY)
# =====================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "deepfashion_review_dataset"
)

CSV_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "deepfashion_metadata_review.csv"
)

random.seed(42)

# =====================================================
# TARGET CATEGORY MAP
# =====================================================

CATEGORY_MAP = {

    # -------------------------
    # TOPS
    # -------------------------

    "Tee": "t-shirt",
    "Jersey": "t-shirt",
    "Henley": "t-shirt",

    "Button-Down": "shirt",
    "Flannel": "shirt",

    "Blouse": "top",
    "Tank": "top",
    "Halter": "top",
    "Poncho": "top",

    "Sweater": "sweater",
    "Turtleneck": "sweater",

    "Hoodie": "hoodie",
    "Cardigan": "cardigan",

    # -------------------------
    # OUTERWEAR
    # -------------------------

    "Jacket": "jacket",
    "Bomber": "jacket",
    "Anorak": "jacket",
    "Parka": "jacket",
    "Kimono": "jacket",

    "Blazer": "coat",
    "Coat": "coat",
    "Peacoat": "coat",
    "Cape": "coat",

    # -------------------------
    # BOTTOMS
    # -------------------------

    "Jeans": "pants",
    "Chinos": "pants",
    "Joggers": "pants",
    "Sweatpants": "pants",
    "Capris": "pants",
    "Culottes": "pants",
    "Gauchos": "pants",
    "Jodhpurs": "pants",

    "Jeggings": "leggings",
    "Leggings": "leggings",

    "Shorts": "shorts",
    "Sweatshorts": "shorts",
    "Cutoffs": "shorts",
    "Trunks": "shorts",

    "Skirt": "skirt",
    "Sarong": "skirt",

    # -------------------------
    # ONE PIECE
    # -------------------------

    "Dress": "dress",
    "Sundress": "dress",
    "Shirtdress": "dress",
    "Nightdress": "dress",
    "Caftan": "dress",
    "Kaftan": "dress",
    "Coverup": "dress",
    "Robe": "dress",

    "Jumpsuit": "jumpsuit",

    "Romper": "romper",
    "Onesie": "romper"
}


# =====================================================
# LOAD VALID CATEGORIES
# =====================================================

print(
    "\nLoading category labels..."
)

valid_categories = set(
    CATEGORY_MAP.keys()
)

print(
    f"Loaded "
    f"{len(valid_categories)} "
    f"supported categories"
)

# =====================================================
# LOAD IMAGE CATEGORY DATA
# =====================================================

print(
    "\nReading image labels..."
)

records = []

with open(
    CATEGORY_IMG_FILE,
    "r",
    encoding="utf-8"
) as f:

    lines = f.readlines()

for line in lines:

    line = line.strip()

    if not line.startswith(
        "img/"
    ):
        continue

    parts = line.split()

    if len(parts) < 2:
        continue

    image_path = parts[0]

    # extract category
    folder_name = os.path.basename(
        os.path.dirname(
            image_path
        )
    )

    category_name = (
        folder_name
        .split("_")[-1]
    )

    if (
        category_name
        not in CATEGORY_MAP
    ):
        continue

    mapped_category = (
        CATEGORY_MAP[
            category_name
        ]
    )

    records.append({

        "path":
        image_path,

        "category":
        mapped_category,

        "original_category":
        category_name
    })

df = pd.DataFrame(
    records
)

print(
    f"\nFiltered images: "
    f"{len(df)}"
)

# =====================================================
# BALANCE DATASET
# =====================================================

# =====================================================
# BALANCE DATASET
# =====================================================

print(
    "\nBalancing dataset..."
)

MAX_PER_CATEGORY = 120

balanced_parts = []

for category in sorted(
    df["category"].unique()
):

    subset = df[
        df["category"]
        == category
    ]

    sample_size = min(
        len(subset),
        MAX_PER_CATEGORY
    )

    subset = subset.sample(
        n=sample_size,
        random_state=42
    )

    balanced_parts.append(
        subset
    )

final_df = pd.concat(
    balanced_parts
)

print(
    "\nFinal category counts:"
)

print(
    final_df["category"]
    .value_counts()
)

# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================

Path(
    OUTPUT_DIR
).mkdir(
    exist_ok=True
)

# =====================================================
# COPY IMAGES
# =====================================================

print(
    "\nCopying images..."
)

metadata_rows = []

for _, row in (
    final_df.iterrows()
):

    relative_path = (
        row["path"]
    )

    source_path = os.path.join(
        DEEPFASHION_ROOT,
        relative_path
    )

    if not os.path.exists(
        source_path
    ):

        print(
            "\nMissing file:"
        )

        print(
            source_path
        )

        continue

    folder_name = os.path.basename(
        os.path.dirname(
            relative_path
        )
    )

    original_name = os.path.basename(
        relative_path
    )

    filename = (
        folder_name
        + "_"
        + original_name
    )

    destination = os.path.join(
        OUTPUT_DIR,
        filename
    )

    shutil.copy2(
        source_path,
        destination
    )

    image_id = os.path.splitext(
        filename
    )[0]

    metadata_rows.append({

        "image_id":
        image_id,

        "category":
        row["category"],

        "style":
        "unknown",

        "silhouette":
        "unknown",

        "neckline":
        "unknown",

        "sleeve":
        "unknown",

        "pattern":
        "unknown",

        "source":
        "deepfashion"
    })

# =====================================================
# SAVE CSV
# =====================================================

metadata_df = pd.DataFrame(
    metadata_rows
)

metadata_df.to_csv(
    CSV_OUTPUT,
    index=False
)

print("\nDONE")
print(
    f"Images copied: "
    f"{len(metadata_df)}"
)

print(
    f"CSV saved to:\n"
    f"{CSV_OUTPUT}"
)