"""Dataset and metadata utilities for the fashion retrieval app.

Usage examples:
    python -m scripts.dataset_tools --help
    python -m scripts.dataset_tools build-review-dataset
    python -m scripts.dataset_tools enrich-metadata
"""

import os
import json

import argparse
import csv
import random
import shutil
import textwrap
from collections import Counter
from io import StringIO
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEEPFASHION_ROOT = PROJECT_ROOT / "DeepFashion"
BENCHMARK_ROOT = DEEPFASHION_ROOT / "Category and Attribute Prediction Benchmark"
ANNO_COARSE = BENCHMARK_ROOT / "Anno_coarse"

CATEGORY_IMG_FILE = ANNO_COARSE / "list_category_img.txt"
ATTR_IMG_FILE = ANNO_COARSE / "list_attr_img.txt"
ATTR_CLOTH_FILE = ANNO_COARSE / "list_attr_cloth.txt"

DEFAULT_REVIEW_DIR = PROJECT_ROOT / "deepfashion_review_dataset"
DEFAULT_REVIEW_CSV = PROJECT_ROOT / "deepfashion_metadata_review.csv"
DEFAULT_ENRICHED_CSV = PROJECT_ROOT / "deepfashion_metadata_enriched.csv"

DEFAULT_METADATA = PROJECT_ROOT / "metadata.csv"
DEFAULT_METADATA_V2 = PROJECT_ROOT / "metadata_v2.csv"

DEFAULT_IMAGES_DIR = PROJECT_ROOT / "dataset" / "images"
DEFAULT_AUDIT_DIR = Path("C:/tmp/custom-metadata-audit")


def sync_taxonomy(
    metadata_path="metadata.csv",
    taxonomy_path="taxonomy.json"
):
    """
    Extract unique labels from metadata.csv
    and save them into taxonomy.json
    """

    print("Reading metadata...")

    df = pd.read_csv(metadata_path)

    taxonomy_columns = [
        "category",
        "silhouette",
        "sleeve",
        "neckline",
        "color",
        "style",
        "pattern"
    ]

    taxonomy = {}

    for column in taxonomy_columns:

        if column not in df.columns:
            print(f"Skipping missing column: {column}")
            continue

        values = (
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        unique_values = sorted(
            [
                value
                for value in values.unique()
                if value.lower() != "none"
                and value != ""
            ]
        )

        taxonomy[column] = unique_values

    # Add unknown as fallback
    for key in taxonomy:
        taxonomy[key].append("none")

    with open(
        taxonomy_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            taxonomy,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Taxonomy saved to {taxonomy_path}"
    )


CATEGORY_MAP = {
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
    "Jacket": "jacket",
    "Bomber": "jacket",
    "Anorak": "jacket",
    "Parka": "jacket",
    "Kimono": "jacket",
    "Blazer": "coat",
    "Coat": "coat",
    "Peacoat": "coat",
    "Cape": "coat",
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
    "Onesie": "romper",
}

SLEEVE_HINTS = [
    "sleeve",
    "sleeveless",
    "batwing",
    "raglan",
    "dolman",
    "flutter",
    "cap-sleeve",
    "bell-sleeve",
    "cuffed",
]

NECKLINE_HINTS = [
    "neck",
    "collar",
    "halter",
    "strapless",
    "sweetheart",
    "square",
    "asymmetrical",
]

PATTERN_HINTS = [
    "print",
    "stripe",
    "striped",
    "floral",
    "paisley",
    "plaid",
    "checked",
    "animal",
    "polka",
    "geo",
    "abstract",
    "denim",
    "lace",
    "textured",
    "embroidered",
]

STYLE_HINTS = [
    "boho",
    "sporty",
    "athletic",
    "basic",
    "vintage",
    "romantic",
    "formal",
    "classic",
    "minimal",
    "street",
    "glam",
    "tailored",
]

SILHOUETTE_HINTS = [
    "a-line",
    "asymmetric",
    "asymmetrical",
    "babydoll",
    "bandage",
    "batwing",
    "bodycon",
    "boxy",
    "cropped",
    "fit",
    "flare",
    "flared",
    "flowy",
    "maxi",
    "midi",
    "mini",
    "oversized",
    "peplum",
    "pleated",
    "skinny",
    "slim",
    "straight",
    "trapeze",
    "wrap",
    "wide",
    "high-low",
    "bodycon",
    "shift",
    "skater",
    "structured",
    "fitted",
]

REFINED_LABELS = """image_id,category,silhouette,sleeve,neckline,style
img_drs_a_001,dress,fit and flare,sleeveless,halter,classic
img_drs_a_002,dress,fit and flare,sleeveless,high neck,minimal
img_drs_a_003,dress,bodycon,sleeveless,v-neck,glam
img_drs_a_004,dress,straight,sleeveless,cowl neck,glam
img_drs_a_005,dress,bodycon,sleeveless,v-neck,bohemian
img_drs_a_006,dress,fit and flare,sleeveless,halter,modern
img_drs_a_007,dress,bodycon,sleeveless,halter,glam
img_drs_a_008,dress,fit and flare,sleeveless,sweetheart,glam
img_drs_a_009,dress,a-line,sleeveless,asymmetrical,minimal
img_drs_a_010,dress,bodycon,long sleeve,round neck,minimal
img_pnt_a_011,pants,wide-leg,none,none,casual
img_pnt_a_012,pants,wide-leg,none,none,sporty
img_pnt_a_013,pants,wide-leg,none,none,casual
img_pnt_a_014,pants,wide-leg,none,none,formal
img_pnt_a_015,pants,wide-leg,none,none,formal
img_pnt_a_016,pants,wide-leg,none,none,formal
img_pnt_a_017,shorts,regular fit,none,none,minimal
img_skt_a_018,skirt,straight,none,none,minimal
img_skt_a_019,skirt,pleated,none,none,formal
img_skt_a_020,skirt,straight,none,none,tailored
img_skt_a_021,skirt,straight,none,none,casual
img_skt_a_022,skirt,straight,none,none,casual
img_skt_a_023,skirt,a-line,none,none,tailored
img_skt_a_024,skirt,pleated,none,none,glam
img_skt_a_025,skirt,straight,none,none,glam
img_skt_a_026,skirt,straight,none,none,glam
img_skt_a_027,skirt,straight,none,none,glam
img_top_a_028,top,fitted,sleeveless,halter,minimal
img_top_a_029,top,fitted,sleeveless,round neck,casual
img_top_a_030,t-shirt,regular fit,short sleeve,round neck,casual
img_top_a_031,top,fitted,sleeveless,square,casual
img_top_a_032,top,fitted,sleeveless,square,modern
img_top_a_033,top,fitted,sleeveless,square,casual
img_top_a_034,top,cropped,off-shoulder,square,cottagecore
img_top_a_035,top,fitted,sleeveless,round neck,casual
img_top_a_036,top,fitted,sleeveless,asymmetrical,modern
img_top_a_037,top,cropped,short sleeve,square,casual
img_top_a_038,t-shirt,fitted,short sleeve,round neck,casual
img_top_a_039,top,fitted,long sleeve,sweetheart,modern
img_top_a_040,top,fitted,sleeveless,collared,casual
img_top_a_041,top,fitted,sleeveless,square,modern
img_top_a_042,top,fitted,off-shoulder,square,modern
img_top_a_043,top,fitted,sleeveless,collared,classic
img_top_a_044,top,fitted,short sleeve,square,modern
img_top_a_045,top,fitted,sleeveless,square,modern
img_top_a_046,top,fitted,sleeveless,square,formal
img_top_a_047,top,fitted,sleeveless,halter,modern
img_top_a_048,top,fitted,sleeveless,halter,modern
img_top_a_049,top,fitted,sleeveless,high neck,modern
img_top_a_050,top,fitted,sleeveless,cowl neck,modern
img_top_a_051,top,fitted,sleeveless,asymmetrical,modern
img_top_a_052,top,peplum,sleeveless,v-neck,casual
img_top_a_053,top,fitted,sleeveless,sweetheart,modern
img_top_a_054,top,fitted,short sleeve,v-neck,modern
img_top_a_055,top,fitted,sleeveless,halter,glam
img_top_a_056,top,peplum,sleeveless,sweetheart,modern
img_top_a_057,top,peplum,sleeveless,sweetheart,glam
img_top_a_058,top,peplum,sleeveless,v-neck,modern
img_top_a_059,top,peplum,sleeveless,v-neck,modern
img_top_a_060,top,peplum,sleeveless,v-neck,glam
img_drs_b_001,dress,a-line,long sleeve,high neck,minimal
img_drs_b_002,dress,straight,sleeveless,asymmetrical,formal
img_drs_b_003,dress,straight,short sleeve,round neck,modern
img_drs_b_004,dress,straight,long sleeve,square,bohemian
img_drs_b_005,dress,a-line,puff sleeve,square,cottagecore
img_drs_b_006,dress,a-line,puff sleeve,square,cottagecore
img_drs_b_007,dress,a-line,sleeveless,v-neck,romantic
img_drs_b_008,dress,bodycon,long sleeve,round neck,formal
img_drs_b_009,jumpsuit,straight,sleeveless,round neck,formal
img_drs_b_010,dress,a-line,puff sleeve,sweetheart,cottagecore
img_drs_b_011,dress,a-line,puff sleeve,v-neck,cottagecore
img_drs_b_012,dress,a-line,puff sleeve,v-neck,cottagecore
img_drs_b_013,dress,a-line,puff sleeve,round neck,cottagecore
img_drs_b_014,dress,straight,sleeveless,halter,bohemian
img_drs_b_015,dress,wrap,sleeveless,v-neck,formal
img_drs_b_016,dress,bodycon,sleeveless,asymmetrical,glam
img_drs_b_017,dress,straight,sleeveless,v-neck,minimal
img_drs_b_018,dress,a-line,long sleeve,round neck,avant garde
img_jkt_b_019,jacket,oversized,long sleeve,collared,casual
img_jkt_b_020,jacket,oversized,long sleeve,collared,casual
img_jkt_b_021,coat,oversized,long sleeve,collared,tailored
img_jkt_b_022,coat,regular fit,long sleeve,collared,tailored
img_jkt_b_023,jacket,regular fit,long sleeve,collared,casual
img_jkt_b_024,jacket,oversized,long sleeve,collared,casual
img_jkt_b_025,jacket,regular fit,long sleeve,collared,casual
img_pnt_b_026,pants,wide-leg,none,none,avant garde
img_pnt_b_027,pants,wide-leg,none,none,modern
img_pnt_b_028,pants,wide-leg,none,none,modern
img_pnt_b_029,skirt,a-line,none,none,minimal
img_pnt_b_030,pants,wide-leg,none,none,sporty
img_pnt_b_031,pants,straight,none,none,tailored
img_pnt_b_032,pants,straight,none,none,tailored
img_pnt_b_033,pants,straight,none,none,sporty
img_pnt_b_034,pants,wide-leg,none,none,tailored
img_pnt_b_035,pants,wide-leg,none,none,tailored
img_pnt_b_036,pants,straight,none,none,casual
img_pnt_b_037,pants,straight,none,none,bohemian
img_pnt_b_038,pants,straight,none,none,bohemian
img_pnt_b_039,pants,wide-leg,none,none,bohemian
img_pnt_b_040,pants,wide-leg,none,none,bohemian
img_pnt_b_041,pants,wide-leg,none,none,tailored
img_skt_b_042,skirt,a-line,none,none,minimal
img_skt_b_043,skirt,asymmetrical,none,none,avant garde
img_top_b_044,top,oversized,short sleeve,round neck,avant garde
img_top_b_045,coat,oversized,long sleeve,collared,avant garde
img_top_b_046,coat,fitted,long sleeve,v-neck,tailored
img_top_b_047,shirt,oversized,long sleeve,collared,avant garde
img_top_b_048,coat,fitted,long sleeve,collared,avant garde
img_top_b_049,top,oversized,short sleeve,round neck,avant garde
img_top_b_050,dress,straight,short sleeve,round neck,casual
img_top_b_051,top,fitted,sleeveless,v-neck,tailored
img_top_b_052,top,fitted,long sleeve,round neck,casual
img_top_b_053,coat,oversized,long sleeve,v-neck,tailored
img_top_b_054,top,regular fit,puff sleeve,round neck,casual
img_top_b_055,shirt,regular fit,long sleeve,collared,cottagecore
img_top_b_056,coat,oversized,long sleeve,v-neck,tailored
img_top_b_057,coat,oversized,long sleeve,v-neck,tailored
img_top_b_058,coat,fitted,long sleeve,v-neck,tailored
img_top_b_059,coat,peplum,long sleeve,high neck,tailored
img_top_b_060,coat,cropped,long sleeve,v-neck,tailored
img_drs_c_001,dress,a-line,short sleeve,v-neck,bohemian
img_drs_c_002,dress,shift,sleeveless,round neck,minimal
img_drs_c_003,dress,straight,long sleeve,round neck,classic
img_drs_c_004,dress,straight,off-shoulder,asymmetrical,modern
img_drs_c_005,dress,straight,long sleeve,v-neck,bohemian
img_drs_c_006,dress,straight,puff sleeve,v-neck,formal
img_drs_c_007,dress,straight,off-shoulder,square,formal
img_drs_c_008,dress,bodycon,off-shoulder,asymmetrical,glam
img_drs_c_009,dress,bodycon,off-shoulder,asymmetrical,glam
img_drs_c_010,dress,fitted,off-shoulder,sweetheart,formal
img_drs_c_011,dress,bodycon,sleeveless,strapless,glam
img_drs_c_012,dress,fit and flare,sleeveless,sweetheart,formal
img_drs_c_013,dress,bodycon,sleeveless,sweetheart,glam
img_drs_c_014,dress,straight,short sleeve,v-neck,casual
img_drs_c_015,dress,bodycon,sleeveless,high neck,avant garde
img_drs_c_016,dress,straight,long sleeve,round neck,avant garde
img_drs_c_017,dress,straight,long sleeve,high neck,casual
img_drs_c_018,dress,asymmetrical,sleeveless,halter,avant garde
img_drs_c_019,dress,straight,sleeveless,asymmetrical,modern
img_skt_c_020,skirt,straight,none,none,avant garde
img_skt_c_021,skirt,straight,none,none,casual
img_skt_c_022,skirt,straight,none,none,sporty
img_skt_c_023,skirt,a-line,none,none,bohemian
img_skt_c_024,skirt,a-line,none,none,bohemian
img_skt_c_025,skirt,pleated,none,none,sporty
img_skt_c_026,shorts,straight,none,none,tailored
img_skt_c_027,skirt,a-line,none,none,bohemian
img_skt_c_028,skirt,a-line,none,none,casual
img_jkt_c_029,jacket,regular fit,long sleeve,collared,avant garde
img_jkt_c_030,jacket,regular fit,long sleeve,collared,casual
img_jkt_c_031,jacket,regular fit,long sleeve,collared,casual
img_jkt_c_032,jacket,regular fit,long sleeve,collared,casual
img_jkt_c_033,jumpsuit,straight,long sleeve,collared,sporty
img_jkt_c_034,jacket,regular fit,long sleeve,collared,casual
img_jkt_c_035,jacket,regular fit,long sleeve,collared,sporty
img_jkt_c_036,jacket,regular fit,long sleeve,high neck,sporty
img_jkt_c_037,jacket,regular fit,long sleeve,collared,avant garde
img_jkt_c_038,jacket,regular fit,long sleeve,collared,minimal
img_jkt_c_039,jacket,fitted,long sleeve,collared,avant garde
img_jkt_c_040,coat,oversized,long sleeve,high neck,sporty
img_jkt_c_041,jacket,oversized,long sleeve,high neck,avant garde
img_jkt_c_042,jacket,fitted,long sleeve,high neck,avant garde
img_jkt_c_043,jacket,oversized,long sleeve,high neck,casual
img_jkt_c_044,jacket,regular fit,long sleeve,collared,casual
img_jkt_c_045,jacket,oversized,long sleeve,collared,casual
img_jkt_c_046,jacket,oversized,long sleeve,high neck,casual
img_pnt_c_047,pants,straight,none,none,formal
img_pnt_c_048,pants,straight,none,none,sporty
img_pnt_c_049,pants,wide-leg,none,none,formal
img_pnt_c_050,pants,wide-leg,none,none,formal
img_pnt_c_051,pants,straight,none,none,casual
img_pnt_c_052,pants,wide-leg,none,none,casual
img_pnt_c_053,pants,wide-leg,none,none,formal
img_pnt_c_054,pants,straight,none,none,casual
img_top_c_055,top,fitted,sleeveless,round neck,casual
img_top_c_056,top,fitted,sleeveless,sweetheart,bohemian
img_top_c_057,top,fitted,sleeveless,strapless,minimal
img_top_c_058,top,cropped,puff sleeve,high neck,avant garde
img_top_c_059,top,fitted,sleeveless,square,bohemian
img_top_c_060,sweater,cropped,long sleeve,round neck,casual
"""


# Reads DeepFashion attribute names from the benchmark annotations.
def load_attribute_names():
    """Load DeepFashion attribute names from the benchmark file."""
    with ATTR_CLOTH_FILE.open("r", encoding="utf-8") as file:
        lines = file.readlines()[2:]

    attribute_names = []
    for line in lines:
        attr_name = " ".join(line.split()[:-1]).strip().lower()
        attribute_names.append(attr_name)

    return attribute_names


# Iterates image attributes from DeepFashion; depends on attribute annotations.
def iter_image_attributes(attribute_names=None):
    """Yield (image_path, active_attrs) from DeepFashion annotations."""
    if attribute_names is None:
        attribute_names = load_attribute_names()

    with ATTR_IMG_FILE.open("r", encoding="utf-8") as file:
        lines = file.readlines()[2:]

    for line in lines:
        parts = line.split()
        image_path = parts[0]
        values = list(map(int, parts[1:]))
        active_attrs = [
            attribute_names[i]
            for i, value in enumerate(values)
            if value == 1
        ]
        yield image_path, active_attrs


# Builds the dataset image_id from a DeepFashion image path.
def image_id_from_image_path(image_path):
    """Convert a DeepFashion image path into the dataset image_id."""
    folder_name = Path(image_path).parent.name
    filename = Path(image_path).name
    return f"{folder_name}_{filename}".replace(".jpg", "")


# Normalizes text for consistent rule-based matching.
def normalize_text(value):
    """Normalize free-form text for fuzzy rule matching."""
    return (
        str(value)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


# Provides a stable fallback label when an inferred label is too sparse.
def sparse_label_fallback(column, category):
    """Return a stable fallback label when a class is too sparse."""
    if column == "silhouette":
        if category in [
            "dress",
            "skirt",
            "romper",
            "jumpsuit",
            "pants",
            "leggings",
        ]:
            return "straight"

        return "regular fit"

    if column == "style":
        if category == "coat":
            return "formal"

        return "casual"

    if column == "pattern":
        return "plain"

    return "none"


# Creates a balanced review dataset; depends on DeepFashion images + labels.
def build_review_dataset(max_per_category, output_dir, csv_output):
    """Build a balanced review dataset from DeepFashion.

    Usage:
        python -m scripts.dataset_tools build-review-dataset
        python -m scripts.dataset_tools build-review-dataset --max-per-category 80
    """
    print("\nLoading category labels...")

    valid_categories = set(CATEGORY_MAP.keys())

    print(f"Loaded {len(valid_categories)} supported categories")

    print("\nReading image labels...")

    records = []

    with CATEGORY_IMG_FILE.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if not line.startswith("img/"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        image_path = parts[0]
        folder_name = Path(image_path).parent.name
        category_name = folder_name.split("_")[-1]

        if category_name not in CATEGORY_MAP:
            continue

        mapped_category = CATEGORY_MAP[category_name]

        records.append(
            {
                "path": image_path,
                "category": mapped_category,
                "original_category": category_name,
            }
        )

    df = pd.DataFrame(records)

    print(f"\nFiltered images: {len(df)}")
    print("\nBalancing dataset...")

    balanced_parts = []

    for category in sorted(df["category"].unique()):
        subset = df[df["category"] == category]
        sample_size = min(len(subset), max_per_category)
        subset = subset.sample(n=sample_size, random_state=42)
        balanced_parts.append(subset)

    final_df = pd.concat(balanced_parts)

    print("\nFinal category counts:")
    print(final_df["category"].value_counts())

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nCopying images...")

    metadata_rows = []

    for _, row in final_df.iterrows():
        relative_path = row["path"]
        source_path = DEEPFASHION_ROOT / relative_path

        if not source_path.exists():
            print("\nMissing file:")
            print(source_path)
            continue

        folder_name = Path(relative_path).parent.name
        original_name = Path(relative_path).name
        filename = f"{folder_name}_{original_name}"

        destination = output_dir / filename
        shutil.copy2(source_path, destination)

        image_id = Path(filename).stem

        metadata_rows.append(
            {
                "image_id": image_id,
                "category": row["category"],
                "style": "unknown",
                "silhouette": "unknown",
                "neckline": "unknown",
                "sleeve": "unknown",
                "pattern": "unknown",
                "source": "deepfashion",
            }
        )

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(csv_output, index=False)

    print("\nDONE")
    print(f"Images copied: {len(metadata_df)}")
    print(f"CSV saved to:\n{csv_output}")


# Prints candidate keywords by attribute group; depends on DeepFashion annotations.
def discover_keywords(min_freq):
    """Print candidate keywords for sleeve/neckline/pattern/style.

    Usage:
        python -m scripts.dataset_tools discover-keywords --min-freq 10
    """
    attribute_names = load_attribute_names()
    sleeve_counter = Counter()
    neckline_counter = Counter()
    pattern_counter = Counter()
    style_counter = Counter()

    for _, active_attrs in iter_image_attributes(attribute_names):
        for attr in active_attrs:
            if any(x in attr for x in SLEEVE_HINTS):
                sleeve_counter[attr] += 1

            if any(x in attr for x in NECKLINE_HINTS):
                neckline_counter[attr] += 1

            if any(x in attr for x in PATTERN_HINTS):
                pattern_counter[attr] += 1

            if any(x in attr for x in STYLE_HINTS):
                style_counter[attr] += 1

    def print_valid(title, counter):
        print("\n" + "=" * 60)
        print(title.upper())
        print("=" * 60)

        for key, value in counter.most_common():
            if value >= min_freq:
                print(f"{key:<35} {value}")

    print(f"Loaded {len(attribute_names)} attributes")

    print_valid("Sleeve", sleeve_counter)
    print_valid("Neckline", neckline_counter)
    print_valid("Pattern", pattern_counter)
    print_valid("Style", style_counter)


# Prints silhouette-related attribute frequencies; depends on DeepFashion annotations.
def discover_silhouettes(min_freq):
    """Print silhouette-related attribute frequencies.

    Usage:
        python -m scripts.dataset_tools discover-silhouettes --min-freq 10
    """
    attribute_names = load_attribute_names()
    counter = Counter()

    for _, active_attrs in iter_image_attributes(attribute_names):
        for attr in active_attrs:
            if any(x in attr for x in SILHOUETTE_HINTS):
                counter[attr] += 1

    print("\nSILHOUETTES")
    print("=" * 60)
    for key, value in counter.most_common():
        if value >= min_freq:
            print(f"{key:<35} {value}")


# Infers metadata attributes; depends on review CSV + DeepFashion annotations.
def enrich_metadata(input_csv, output_csv):
    """Infer attributes and write an enriched metadata CSV.

    Usage:
        python -m scripts.dataset_tools enrich-metadata
        python -m scripts.dataset_tools enrich-metadata --input-csv deepfashion_metadata_review.csv
    """
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")

    attribute_lookup = {}
    attribute_names = load_attribute_names()
    for image_path, active_attrs in iter_image_attributes(attribute_names):
        image_id = image_id_from_image_path(image_path)
        attribute_lookup[image_id] = active_attrs

    print(f"Loaded {len(attribute_lookup)} image attributes")

    sleeves = []
    necklines = []
    patterns = []
    silhouettes = []
    styles = []

    for _, row in df.iterrows():
        image_id = row["image_id"]
        attrs = attribute_lookup.get(image_id, [])

        filename_text = normalize_text(image_id)
        attrs_text = normalize_text(" ".join(attrs))
        combined_text = f"{filename_text} {attrs_text}"
        attrs_text = combined_text

        category = str(row["category"]).lower()

        sleeve = "none"

        if "sleeveless" in combined_text:
            sleeve = "sleeveless"

        elif any(x in combined_text for x in ["raglan sleeve", "raglan"]):
            sleeve = "raglan sleeve"

        elif any(x in combined_text for x in ["dolman sleeve", "dolman-sleeve", "dolman"]):
            sleeve = "dolman sleeve"

        elif any(x in combined_text for x in ["flutter sleeve", "flutter-sleeve", "flutter"]):
            sleeve = "flutter sleeve"

        elif "bell sleeve" in combined_text:
            sleeve = "bell sleeve"

        elif "cap sleeve" in combined_text:
            sleeve = "cap sleeve"

        elif any(x in combined_text for x in ["cuffed sleeve", "cuffed-sleeve", "cuffed"]):
            sleeve = "cuffed sleeve"

        elif "batwing" in combined_text:
            sleeve = "batwing"

        elif "lace sleeve" in combined_text:
            sleeve = "lace sleeve"

        elif "drop sleeve" in combined_text:
            sleeve = "drop sleeve"

        elif any(x in combined_text for x in ["long sleeve", "long-sleeve", "long-sleeved"]):
            sleeve = "long sleeve"

        elif "sleeve" in combined_text:
            sleeve = "short sleeve"

        if category in ["pants", "leggings", "shorts", "skirt"]:
            sleeve = "none"

        sleeves.append(sleeve)

        neckline = "none"

        if any(x in combined_text for x in ["arrow collar", "notched collar", "collared", "collar"]):
            neckline = "collared"

        elif "collarless" in attrs_text:
            neckline = "collarless"

        elif "deep v neck" in attrs_text or "v neck" in attrs_text:
            neckline = "v-neck"

        elif "crew neck" in attrs_text:
            neckline = "crew neck"

        elif "boat neck" in attrs_text:
            neckline = "boat neck"

        elif "cowl neck" in attrs_text:
            neckline = "cowl neck"

        elif any(x in combined_text for x in ["mock neck", "mock-neck"]):
            neckline = "mock neck"

        elif "high neck" in combined_text:
            neckline = "high neck"

        elif "scoop neck" in combined_text:
            neckline = "scoop neck"

        elif "split neck" in combined_text:
            neckline = "split neck"

        elif "tie neck" in combined_text:
            neckline = "tie neck"

        elif "turtle neck" in combined_text:
            neckline = "turtle neck"

        elif "halter" in attrs_text:
            neckline = "halter"

        elif "strapless" in attrs_text:
            neckline = "strapless"

        elif "sweetheart" in attrs_text:
            neckline = "sweetheart"

        elif "square" in attrs_text:
            neckline = "square"

        elif "illusion neckline" in attrs_text:
            neckline = "illusion neckline"

        elif "asymmetrical" in attrs_text:
            neckline = "asymmetrical"

        if "halter" in filename_text:
            neckline = "halter"

        elif "strapless" in filename_text:
            neckline = "strapless"

        elif "v neck" in filename_text:
            neckline = "v-neck"

        elif "crew neck" in filename_text:
            neckline = "crew neck"

        elif "cowl neck" in filename_text:
            neckline = "cowl neck"

        if category in ["pants", "leggings", "shorts", "skirt"]:
            neckline = "none"

        necklines.append(neckline)

        pattern = "plain"

        if "lace" in attrs_text:
            pattern = "lace"

        elif "botanical print" in attrs_text:
            pattern = "botanical print"

        elif "folk print" in attrs_text:
            pattern = "folk print"

        elif "baroque print" in attrs_text:
            pattern = "baroque print"

        elif "pinstripe" in attrs_text:
            pattern = "pinstripe"

        elif any(x in combined_text for x in ["nautical stripe", "nautical striped"]):
            pattern = "nautical stripe"

        elif "floral" in combined_text:
            pattern = "floral"

        elif any(x in attrs_text for x in ["stripe", "striped", "stripes"]):
            pattern = "striped"

        elif any(x in attrs_text for x in ["checked", "plaid"]):
            pattern = "checkered"

        elif "paisley" in combined_text:
            pattern = "paisley"

        elif any(x in attrs_text for x in ["animal", "leopard", "giraffe", "elephant"]):
            pattern = "animal print"

        elif any(x in attrs_text for x in ["geo", "geometric"]):
            pattern = "geometric"

        elif "polka" in combined_text:
            pattern = "polka dot"

        elif "embroidered" in combined_text:
            pattern = "embroidered"

        elif "denim" in combined_text:
            pattern = "denim"

        elif any(x in attrs_text for x in ["textured", "knit", "ribbed", "georgette"]):
            pattern = "textured"

        elif "abstract" in combined_text:
            pattern = "abstract"

        elif any(x in attrs_text for x in ["print", "printed"]):
            pattern = "graphic print"

        if "denim" in filename_text:
            pattern = "denim"

        elif any(x in filename_text for x in ["checked", "plaid"]):
            pattern = "checkered"

        elif "floral" in filename_text:
            pattern = "floral"

        elif any(x in filename_text for x in ["stripe", "striped", "stripes"]):
            pattern = "striped"

        elif "paisley" in filename_text:
            pattern = "paisley"

        elif "abstract" in filename_text:
            pattern = "abstract"

        patterns.append(pattern)

        if category in ["dress", "skirt", "romper", "jumpsuit"]:
            silhouette = "straight"

        elif category in ["pants", "leggings"]:
            silhouette = "straight"

        else:
            silhouette = "regular fit"

        if category in ["dress", "skirt", "romper", "jumpsuit"]:
            if "bodycon" in attrs_text:
                silhouette = "bodycon"

            elif "fit flare" in attrs_text:
                silhouette = "fit and flare"

            elif "skater" in attrs_text:
                silhouette = "skater"

            elif "shift" in attrs_text:
                silhouette = "shift"

            elif "a line" in attrs_text:
                silhouette = "a-line"

            elif "babydoll" in attrs_text:
                silhouette = "babydoll"

            elif "trapeze" in attrs_text:
                silhouette = "trapeze"

            elif "peplum" in attrs_text:
                silhouette = "peplum"

            elif "wrap" in attrs_text:
                silhouette = "wrap"

            elif any(x in attrs_text for x in ["asymmetrical", "asymmetric"]):
                silhouette = "asymmetrical"

            elif "high low" in attrs_text:
                silhouette = "high-low"

            elif "flowy" in attrs_text:
                silhouette = "flowy"

            elif "pleated" in attrs_text:
                silhouette = "pleated"

            elif "mini" in attrs_text:
                silhouette = "mini"

            elif "midi" in attrs_text:
                silhouette = "midi"

            elif "maxi" in attrs_text:
                silhouette = "maxi"

        elif category in [
            "top",
            "shirt",
            "t-shirt",
            "hoodie",
            "cardigan",
            "sweater",
            "jacket",
            "coat",
        ]:
            if "oversized" in attrs_text:
                silhouette = "oversized"

            elif "boxy" in attrs_text:
                silhouette = "boxy"

            elif "cropped" in attrs_text:
                silhouette = "cropped"

            elif "peplum" in attrs_text:
                silhouette = "peplum"

            elif "batwing" in attrs_text:
                silhouette = "batwing"

            elif "fitted" in attrs_text:
                silhouette = "fitted"

            elif any(x in attrs_text for x in ["asymmetrical", "asymmetric"]):
                silhouette = "asymmetrical"

            elif "wrap" in attrs_text:
                silhouette = "wrap"

            elif "structured" in attrs_text:
                silhouette = "structured"

        elif category in ["pants", "leggings"]:
            if "skinny" in attrs_text:
                silhouette = "skinny"

            elif "slim" in attrs_text:
                silhouette = "slim"

            elif "straight leg" in attrs_text:
                silhouette = "straight leg"

            elif "wide leg" in attrs_text:
                silhouette = "wide leg"

            elif any(x in attrs_text for x in ["flare", "flared"]):
                silhouette = "flared"

            elif "cropped" in attrs_text:
                silhouette = "cropped"

        silhouettes.append(silhouette)

        style = "casual"

        if any(x in attrs_text for x in ["athletic", "sporty"]):
            style = "sporty"

        elif "boho" in combined_text:
            style = "bohemian"

        elif "basic" in combined_text:
            style = "minimal"

        elif "classic" in attrs_text:
            style = "classic"

        elif category == "coat":
            style = "formal"

        elif category == "hoodie" and pattern == "graphic print":
            style = "streetwear"

        elif category == "dress" and pattern in ["lace", "floral", "embroidered"]:
            style = "romantic"

        styles.append(style)

    df["sleeve"] = sleeves
    df["neckline"] = necklines
    df["pattern"] = patterns
    df["silhouette"] = silhouettes
    df["style"] = styles

    for column in ["silhouette", "sleeve", "neckline", "pattern", "style"]:
        counts = Counter(df[column])
        valid = {key for key, value in counts.items() if value >= 5}

        df[column] = df[column].apply(lambda value: value if value in valid else None)

        df[column] = [
            value
            if pd.notna(value)
            else sparse_label_fallback(column, str(category).lower())
            for value, category in zip(df[column], df["category"])
        ]

    df.to_csv(output_csv, index=False)

    print("\nDONE")
    print(f"Saved to:\n{output_csv}")
    print("\nColumn stats:")

    for col in ["silhouette", "style", "neckline", "sleeve", "pattern"]:
        print("\n", col.upper())
        print(df[col].value_counts())


# Merges enriched metadata into the main catalog CSV.
def merge_metadata(old_csv, new_csv, output_csv):
    """Merge enriched metadata into the main metadata CSV.

    Usage:
        python -m scripts.dataset_tools merge-metadata
        python -m scripts.dataset_tools merge-metadata --new-csv deepfashion_metadata_enriched.csv
    """
    old_df = pd.read_csv(old_csv)
    new_df = pd.read_csv(new_csv)

    print(f"Old rows: {len(old_df)}")
    print(f"New rows: {len(new_df)}")

    merged_df = pd.concat([old_df, new_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset="image_id")

    merged_df.to_csv(output_csv, index=False)

    print("\nDONE")
    print(f"Final rows: {len(merged_df)}")
    print("\nCategory counts:")
    print(merged_df["category"].value_counts())


# Renders audit sheets for manual review; depends on metadata CSV + image files.
def make_custom_metadata_audit_sheets(
    metadata_path,
    image_dir,
    output_dir,
    batch_size,
    columns,
    rows,
    cell_width,
    cell_height,
):
    """Generate audit sheets for manual metadata review.

    Usage:
        python -m scripts.dataset_tools audit-sheets
        python -m scripts.dataset_tools audit-sheets --output-dir C:/tmp/custom-metadata-audit
    """
    from PIL import Image, ImageDraw, ImageFont

    output_dir.mkdir(parents=True, exist_ok=True)

    with metadata_path.open(newline="", encoding="utf-8-sig") as file:
        metadata = list(csv.DictReader(file))

    font = ImageFont.load_default()

    def draw_wrapped(draw, text, x, y):
        for line in textwrap.wrap(text, width=60):
            draw.text((x, y), line, fill="black", font=font)
            y += 14
        return y

    for batch_start in range(0, len(metadata), batch_size):
        batch = metadata[batch_start:batch_start + batch_size]
        sheet = Image.new(
            "RGB",
            (columns * cell_width, rows * cell_height),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        for position, row in enumerate(batch):
            x = (position % columns) * cell_width
            y = (position // columns) * cell_height

            with Image.open(image_dir / row["file_name"]) as image:
                image = image.convert("RGB")
                image.thumbnail((300, 300))
                sheet.paste(
                    image,
                    (
                        x + (300 - image.width) // 2 + 5,
                        y + 5,
                    ),
                )

            label_y = y + 312
            labels = [
                f"{batch_start + position + 1:03d} {row['image_id']}",
                f"cat={row['category']} style={row['style']}",
                f"sil={row['silhouette']} sleeve={row['sleeve']}",
                f"neck={row['neckline']} pat={row['pattern']}",
            ]

            for label in labels:
                label_y = draw_wrapped(draw, label, x + 8, label_y)

        output_path = output_dir / f"batch_{batch_start // batch_size + 1:02d}.jpg"
        sheet.save(output_path, quality=94)

    print(f"Generated {(len(metadata) + batch_size - 1) // batch_size} sheets")
    print(output_dir)


# Applies curated labels to metadata.csv and writes metadata_v2.csv.
def apply_refined_labels(input_csv, output_csv):
    """Apply refined labels and write metadata_v2.csv.

    Usage:
        python -m scripts.dataset_tools apply-refined-labels
    """
    metadata = pd.read_csv(input_csv)
    refined = pd.read_csv(
        StringIO(REFINED_LABELS),
        quoting=csv.QUOTE_MINIMAL,
    )

    if len(refined) != len(metadata):
        raise ValueError(
            f"Expected {len(metadata)} reviewed rows, got {len(refined)}."
        )

    if refined["image_id"].duplicated().any():
        raise ValueError("Reviewed labels contain duplicate image IDs.")

    missing_ids = set(metadata["image_id"]) - set(refined["image_id"])
    extra_ids = set(refined["image_id"]) - set(metadata["image_id"])

    if missing_ids or extra_ids:
        raise ValueError(
            "Reviewed image IDs do not match metadata.csv. "
            f"Missing: {sorted(missing_ids)}. Extra: {sorted(extra_ids)}."
        )

    columns_to_refine = [
        "category",
        "silhouette",
        "sleeve",
        "neckline",
        "style",
    ]

    metadata = metadata.drop(columns=columns_to_refine).merge(
        refined,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    metadata = metadata[
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

    metadata.to_csv(output_csv, index=False)

    print(f"Saved {len(metadata)} reviewed rows to {output_csv}")


# Defines the CLI interface for dataset tooling.
def build_parser():
    """Build the CLI parser for dataset tooling subcommands."""
    parser = argparse.ArgumentParser(
        description="Dataset and metadata utilities for the fashion retrieval app."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    review_parser = subparsers.add_parser(
        "build-review-dataset",
        help="Create a balanced DeepFashion review dataset and CSV.",
    )
    review_parser.add_argument(
        "--max-per-category",
        type=int,
        default=120,
    )
    review_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_REVIEW_DIR),
    )
    review_parser.add_argument(
        "--csv-output",
        default=str(DEFAULT_REVIEW_CSV),
    )
    subparsers.add_parser(
        "sync-taxonomy",
        help="Generate taxonomy.json from metadata.csv"
    )

    keywords_parser = subparsers.add_parser(
        "discover-keywords",
        help="Print candidate attribute keywords by category.",
    )
    keywords_parser.add_argument(
        "--min-freq",
        type=int,
        default=5,
    )

    silhouettes_parser = subparsers.add_parser(
        "discover-silhouettes",
        help="Print silhouette attribute frequencies.",
    )
    silhouettes_parser.add_argument(
        "--min-freq",
        type=int,
        default=5,
    )

    enrich_parser = subparsers.add_parser(
        "enrich-metadata",
        help="Enrich DeepFashion review CSV with inferred attributes.",
    )
    enrich_parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_REVIEW_CSV),
    )
    enrich_parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_ENRICHED_CSV),
    )

    merge_parser = subparsers.add_parser(
        "merge-metadata",
        help="Merge enriched DeepFashion metadata into metadata.csv.",
    )
    merge_parser.add_argument(
        "--old-csv",
        default=str(DEFAULT_METADATA),
    )
    merge_parser.add_argument(
        "--new-csv",
        default=str(DEFAULT_ENRICHED_CSV),
    )
    merge_parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_METADATA),
    )

    audit_parser = subparsers.add_parser(
        "audit-sheets",
        help="Generate custom metadata audit sheets.",
    )
    audit_parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA),
    )
    audit_parser.add_argument(
        "--image-dir",
        default=str(DEFAULT_IMAGES_DIR),
    )
    audit_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_AUDIT_DIR),
    )
    audit_parser.add_argument("--batch-size", type=int, default=20)
    audit_parser.add_argument("--columns", type=int, default=4)
    audit_parser.add_argument("--rows", type=int, default=5)
    audit_parser.add_argument("--cell-width", type=int, default=430)
    audit_parser.add_argument("--cell-height", type=int, default=430)

    refined_parser = subparsers.add_parser(
        "apply-refined-labels",
        help="Apply refined labels to metadata.csv and write metadata_v2.csv.",
    )
    refined_parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_METADATA),
    )
    refined_parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_METADATA_V2),
    )

    return parser


# CLI entry point for dataset tooling.
def main():
    """Entry point for dataset tooling CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-review-dataset":
        random.seed(42)
        build_review_dataset(
            max_per_category=args.max_per_category,
            output_dir=Path(args.output_dir),
            csv_output=Path(args.csv_output),
        )
    
    elif args.command == "sync-taxonomy":
        sync_taxonomy()

    elif args.command == "discover-keywords":
        discover_keywords(args.min_freq)

    elif args.command == "discover-silhouettes":
        discover_silhouettes(args.min_freq)

    elif args.command == "enrich-metadata":
        enrich_metadata(
            input_csv=Path(args.input_csv),
            output_csv=Path(args.output_csv),
        )

    elif args.command == "merge-metadata":
        merge_metadata(
            old_csv=Path(args.old_csv),
            new_csv=Path(args.new_csv),
            output_csv=Path(args.output_csv),
        )

    elif args.command == "audit-sheets":
        make_custom_metadata_audit_sheets(
            metadata_path=Path(args.metadata),
            image_dir=Path(args.image_dir),
            output_dir=Path(args.output_dir),
            batch_size=args.batch_size,
            columns=args.columns,
            rows=args.rows,
            cell_width=args.cell_width,
            cell_height=args.cell_height,
        )

    elif args.command == "apply-refined-labels":
        apply_refined_labels(
            input_csv=Path(args.input_csv),
            output_csv=Path(args.output_csv),
        )


    
if __name__ == "__main__":
    main()