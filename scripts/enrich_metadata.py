import os
import pandas as pd
from collections import Counter


# =====================================================
# PATHS
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

ANNO_COARSE = os.path.join(
    DEEPFASHION_ROOT,
    "Category and Attribute Prediction Benchmark",
    "Anno_coarse"
)

ATTR_IMG_FILE = os.path.join(
    ANNO_COARSE,
    "list_attr_img.txt"
)

ATTR_CLOTH_FILE = os.path.join(
    ANNO_COARSE,
    "list_attr_cloth.txt"
)

INPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "deepfashion_metadata_review.csv"
)

OUTPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "deepfashion_metadata_enriched.csv"
)


# =====================================================
# LOAD METADATA
# =====================================================

df = pd.read_csv(
    INPUT_CSV
)

print(
    f"Loaded {len(df)} rows"
)


# =====================================================
# LOAD ATTRIBUTE NAMES
# =====================================================

attribute_names = []

with open(
    ATTR_CLOTH_FILE,
    "r",
    encoding="utf-8"
) as f:

    lines = f.readlines()[2:]

for line in lines:

    attr_name = (
        " ".join(
            line.split()[:-1]
        )
        .strip()
        .lower()
    )

    attribute_names.append(
        attr_name
    )

print(
    f"Loaded "
    f"{len(attribute_names)} "
    f"attributes"
)


# =====================================================
# LOAD IMAGE ATTRIBUTES
# =====================================================

attribute_lookup = {}

with open(
    ATTR_IMG_FILE,
    "r",
    encoding="utf-8"
) as f:

    lines = f.readlines()[2:]

for line in lines:

    parts = line.split()

    image_path = parts[0]

    values = list(
        map(
            int,
            parts[1:]
        )
    )

    active_attrs = []

    for i, value in enumerate(
        values
    ):

        if value == 1:

            active_attrs.append(
                attribute_names[i]
            )

    folder_name = os.path.basename(
        os.path.dirname(
            image_path
        )
    )

    filename = os.path.basename(
        image_path
    )

    image_id = (
        folder_name
        + "_"
        + filename
    ).replace(
        ".jpg",
        ""
    )

    attribute_lookup[
        image_id
    ] = active_attrs


print(
    f"Loaded "
    f"{len(attribute_lookup)} "
    f"image attributes"
)


# =====================================================
# HELPERS
# =====================================================

def has_any(
    attrs,
    candidates
):
    return any(
        x in attrs
        for x in candidates
    )


def contains_any(
    attrs,
    keywords
):
    return any(
        any(
            k in attr
            for k in keywords
        )
        for attr in attrs
    )


def normalize_text(
    value
):
    return (
        str(value)
        .lower()
        .replace(
            "_",
            " "
        )
        .replace(
            "-",
            " "
        )
    )


def sparse_label_fallback(
    column,
    category
):
    if column == "silhouette":
        if category in [
            "dress",
            "skirt",
            "romper",
            "jumpsuit",
            "pants",
            "leggings"
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


# =====================================================
# ENRICH METADATA
# =====================================================

sleeves = []
necklines = []
patterns = []
silhouettes = []
styles = []


for _, row in df.iterrows():

    image_id = row[
        "image_id"
    ]

    attrs = (
        attribute_lookup.get(
            image_id,
            []
        )
    )

    filename_text = normalize_text(
        image_id
    )

    attrs_text = normalize_text(
        " ".join(
            attrs
        )
    )

    combined_text = (
        filename_text
        + " "
        + attrs_text
    )

    attrs_text = combined_text

    category = str(
        row["category"]
    ).lower()

    # =================================================
    # SLEEVE
    # =================================================

    sleeve = "none"

    if (
        "sleeveless"
        in combined_text
    ):
        sleeve = (
            "sleeveless"
        )

    elif any(
        x in combined_text
        for x in [
            "raglan sleeve",
            "raglan"
        ]
    ):
        sleeve = (
            "raglan sleeve"
        )

    elif any(
        x in combined_text
        for x in [
            "dolman sleeve",
            "dolman-sleeve",
            "dolman"
        ]
    ):
        sleeve = (
            "dolman sleeve"
        )

    elif any(
        x in combined_text
        for x in [
            "flutter sleeve",
            "flutter-sleeve",
            "flutter"
        ]
    ):
        sleeve = (
            "flutter sleeve"
        )

    elif (
        "bell sleeve"
        in combined_text
    ):
        sleeve = (
            "bell sleeve"
        )

    elif (
        "cap sleeve"
        in combined_text
    ):
        sleeve = (
            "cap sleeve"
        )

    elif any(
        x in combined_text
        for x in [
            "cuffed sleeve",
            "cuffed-sleeve",
            "cuffed"
        ]
    ):
        sleeve = (
            "cuffed sleeve"
        )

    elif (
        "batwing"
        in combined_text
    ):
        sleeve = (
            "batwing"
        )

    elif (
        "lace sleeve"
        in combined_text
    ):
        sleeve = (
            "lace sleeve"
        )

    elif (
        "drop sleeve"
        in combined_text
    ):
        sleeve = (
            "drop sleeve"
        )

    elif any(
        x in combined_text
        for x in [
            "long sleeve",
            "long-sleeve",
            "long-sleeved"
        ]
    ):
        sleeve = (
            "long sleeve"
        )

    elif (
        "sleeve"
        in combined_text
    ):
        sleeve = (
            "short sleeve"
        )

    if category in [
        "pants",
        "leggings",
        "shorts",
        "skirt"
    ]:
        sleeve = "none"

    sleeves.append(
        sleeve
    )

    # =================================================
    # NECKLINE
    # =================================================

    neckline = "none"

    if any(
        x in combined_text
        for x in [
            "arrow collar",
            "notched collar",
            "collared",
            "collar"
        ]
    ):
        neckline = (
            "collared"
        )

    elif (
        "collarless"
        in attrs_text
    ):
        neckline = (
            "collarless"
        )

    elif (
        "deep v neck"
        in attrs_text
        or "v neck"
        in attrs_text
    ):
        neckline = (
            "v-neck"
        )

    elif (
        "crew neck"
        in attrs_text
    ):
        neckline = (
            "crew neck"
        )

    elif (
        "boat neck"
        in attrs_text
    ):
        neckline = (
            "boat neck"
        )

    elif (
        "cowl neck"
        in attrs_text
    ):
        neckline = (
            "cowl neck"
        )

    elif any(
        x in combined_text
        for x in [
            "mock neck",
            "mock-neck"
        ]
    ):
        neckline = (
            "mock neck"
        )

    elif (
        "high neck"
        in combined_text
    ):
        neckline = (
            "high neck"
        )

    elif (
        "scoop neck"
        in combined_text
    ):
        neckline = (
            "scoop neck"
        )

    elif (
        "split neck"
        in combined_text
    ):
        neckline = (
            "split neck"
        )

    elif (
        "tie neck"
        in combined_text
    ):
        neckline = (
            "tie neck"
        )

    elif (
        "turtle neck"
        in combined_text
    ):
        neckline = (
            "turtle neck"
        )

    elif (
        "halter"
        in attrs_text
    ):
        neckline = (
            "halter"
        )

    elif (
        "strapless"
        in attrs_text
    ):
        neckline = (
            "strapless"
        )

    elif (
        "sweetheart"
        in attrs_text
    ):
        neckline = (
            "sweetheart"
        )

    elif (
        "square"
        in attrs_text
    ):
        neckline = (
            "square"
        )

    elif (
        "illusion neckline"
        in attrs_text
    ):
        neckline = (
            "illusion neckline"
        )

    elif (
        "asymmetrical"
        in attrs_text
    ):
        neckline = (
            "asymmetrical"
        )

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

    if category in [
        "pants",
        "leggings",
        "shorts",
        "skirt"
    ]:
        neckline = "none"

    necklines.append(
        neckline
    )

    # =================================================
    # PATTERN
    # =================================================

    pattern = "plain"

    if "lace" in attrs_text:
        pattern = "lace"

    elif (
        "botanical print"
        in attrs_text
    ):
        pattern = (
            "botanical print"
        )

    elif (
        "folk print"
        in attrs_text
    ):
        pattern = (
            "folk print"
        )

    elif (
        "baroque print"
        in attrs_text
    ):
        pattern = (
            "baroque print"
        )

    elif (
        "pinstripe"
        in attrs_text
    ):
        pattern = (
            "pinstripe"
        )

    elif any(
        x in combined_text
        for x in [
            "nautical stripe",
            "nautical striped"
        ]
    ):
        pattern = (
            "nautical stripe"
        )

    elif (
        "floral"
        in combined_text
    ):
        pattern = (
            "floral"
        )

    elif any(
        x in attrs_text
        for x in [
            "stripe",
            "striped",
            "stripes"
        ]
    ):
        pattern = (
            "striped"
        )

    elif any(
        x in attrs_text
        for x in [
            "checked",
            "plaid"
        ]
    ):
        pattern = (
            "checkered"
        )

    elif (
        "paisley"
        in combined_text
    ):
        pattern = (
            "paisley"
        )

    elif any(
        x in attrs_text
        for x in [
            "animal",
            "leopard",
            "giraffe",
            "elephant"
        ]
    ):
        pattern = (
            "animal print"
        )

    elif any(
        x in attrs_text
        for x in [
            "geo",
            "geometric"
        ]
    ):
        pattern = (
            "geometric"
        )

    elif (
        "polka"
        in combined_text
    ):
        pattern = (
            "polka dot"
        )

    elif (
        "embroidered"
        in combined_text
    ):
        pattern = (
            "embroidered"
        )

    elif (
        "denim"
        in combined_text
    ):
        pattern = (
            "denim"
        )

    elif any(
        x in attrs_text
        for x in [
            "textured",
            "knit",
            "ribbed",
            "georgette"
        ]
    ):
        pattern = (
            "textured"
        )

    elif (
        "abstract"
        in combined_text
    ):
        pattern = (
            "abstract"
        )

    elif any(
        x in attrs_text
        for x in [
            "print",
            "printed"
        ]
    ):
        pattern = (
            "graphic print"
        )

    if "denim" in filename_text:
        pattern = "denim"

    elif any(
        x in filename_text
        for x in [
            "checked",
            "plaid"
        ]
    ):
        pattern = "checkered"

    elif "floral" in filename_text:
        pattern = "floral"

    elif any(
        x in filename_text
        for x in [
            "stripe",
            "striped",
            "stripes"
        ]
    ):
        pattern = "striped"

    elif "paisley" in filename_text:
        pattern = "paisley"

    elif "abstract" in filename_text:
        pattern = "abstract"

    patterns.append(
        pattern
    )

    # =================================================
    # SILHOUETTE
    # =================================================

    if category in [
        "dress",
        "skirt",
        "romper",
        "jumpsuit"
    ]:
        silhouette = "straight"

    elif category in [
        "pants",
        "leggings"
    ]:
        silhouette = "straight"

    else:
        silhouette = "regular fit"
    

    # -----------------------------------------
    # DRESSES / SKIRTS
    # -----------------------------------------

    if category in [
        "dress",
        "skirt",
        "romper",
        "jumpsuit"
    ]:

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

        elif any(
            x in attrs_text
            for x in [
                "asymmetrical",
                "asymmetric"
            ]
        ):
            silhouette = (
                "asymmetrical"
            )

        elif (
            "high low"
            in attrs_text
        ):
            silhouette = (
                "high-low"
            )

        elif (
            "flowy"
            in attrs_text
        ):
            silhouette = (
                "flowy"
            )

        elif (
            "pleated"
            in attrs_text
        ):
            silhouette = (
                "pleated"
            )

        elif (
            "mini"
            in attrs_text
        ):
            silhouette = (
                "mini"
            )

        elif (
            "midi"
            in attrs_text
        ):
            silhouette = (
                "midi"
            )

        elif (
            "maxi"
            in attrs_text
        ):
            silhouette = (
                "maxi"
            )

    # -----------------------------------------
    # TOPS / OUTERWEAR
    # -----------------------------------------

    elif category in [

        "top",
        "shirt",
        "t-shirt",
        "hoodie",
        "cardigan",
        "sweater",
        "jacket",
        "coat"

    ]:

        if (
            "oversized"
            in attrs_text
        ):
            silhouette = (
                "oversized"
            )

        elif (
            "boxy"
            in attrs_text
        ):
            silhouette = (
                "boxy"
            )

        elif (
            "cropped"
            in attrs_text
        ):
            silhouette = (
                "cropped"
            )

        elif (
            "peplum"
            in attrs_text
        ):
            silhouette = (
                "peplum"
            )

        elif (
            "batwing"
            in attrs_text
        ):
            silhouette = (
                "batwing"
            )

        elif (
            "fitted"
            in attrs_text
        ):
            silhouette = (
                "fitted"
            )

        elif any(
            x in attrs_text
            for x in [
                "asymmetrical",
                "asymmetric"
            ]
        ):
            silhouette = (
                "asymmetrical"
            )

        elif (
            "wrap"
            in attrs_text
        ):
            silhouette = (
                "wrap"
            )

        elif (
            "structured"
            in attrs_text
        ):
            silhouette = (
                "structured"
            )

    # -----------------------------------------
    # PANTS / LEGGINGS
    # -----------------------------------------

    elif category in [
        "pants",
        "leggings"
    ]:

        if (
            "skinny"
            in attrs_text
        ):
            silhouette = (
                "skinny"
            )

        elif (
            "slim"
            in attrs_text
        ):
            silhouette = (
                "slim"
            )

        elif (
            "straight leg"
            in attrs_text
        ):
            silhouette = (
                "straight leg"
            )

        elif (
            "wide leg"
            in attrs_text
        ):
            silhouette = (
                "wide leg"
            )

        elif any(
            x in attrs_text
            for x in [
                "flare",
                "flared"
            ]
        ):
            silhouette = (
                "flared"
            )

        elif (
            "cropped"
            in attrs_text
        ):
            silhouette = (
                "cropped"
            )

    silhouettes.append(
        silhouette
    )

    # =================================================
    # STYLE
    # =================================================

    style = "casual"

    if any(
        x in attrs_text
        for x in [
            "athletic",
            "sporty"
        ]
    ):
        style = (
            "sporty"
        )

    elif (
        "boho"
        in combined_text
    ):
        style = (
            "bohemian"
        )

    elif (
        "basic"
        in combined_text
    ):
        style = (
            "minimal"
        )

    elif (
        "classic"
        in attrs_text
    ):
        style = (
            "classic"
        )

    elif (
        category
        == "coat"
    ):
        style = (
            "formal"
        )

    elif (
        category
        == "hoodie"
        and pattern
        == "graphic print"
    ):
        style = (
            "streetwear"
        )

    elif (
        category
        == "dress"
        and pattern in [
            "lace",
            "floral",
            "embroidered"
        ]
    ):
        style = (
            "romantic"
        )

    styles.append(
        style
    )

# =====================================================
# ASSIGN
# =====================================================

df[
    "sleeve"
] = sleeves

df[
    "neckline"
] = necklines

df[
    "pattern"
] = patterns

df[
    "silhouette"
] = silhouettes

df[
    "style"
] = styles


# =====================================================
# REMOVE SPARSE LABELS
# =====================================================

for column in [
    "silhouette",
    "sleeve",
    "neckline",
    "pattern",
    "style"
]:

    counts = Counter(
        df[column]
    )

    valid = {
        k
        for k, v
        in counts.items()
        if v >= 5
    }

    df[column] = df[
        column
    ].apply(

        lambda x:
        x
        if x in valid
        else None
    )

    df[column] = [
        value
        if pd.notna(
            value
        )
        else sparse_label_fallback(
            column,
            str(category).lower()
        )
        for value, category
        in zip(
            df[column],
            df["category"]
        )
    ]


# =====================================================
# SAVE
# =====================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    "\nDONE"
)

print(
    f"Saved to:\n"
    f"{OUTPUT_CSV}"
)

print(
    "\nColumn stats:"
)

for col in [
    "silhouette",
    "style",
    "neckline",
    "sleeve",
    "pattern"
]:

    print(
        "\n",
        col.upper()
    )

    print(
        df[col]
        .value_counts()
    )
