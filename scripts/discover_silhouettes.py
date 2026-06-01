import os
from collections import Counter


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ANNO_COARSE = os.path.join(
    PROJECT_ROOT,
    "DeepFashion",
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


# =====================================================
# LOAD ATTRIBUTES
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


# =====================================================
# HINTS
# =====================================================

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
    "fitted"
]


# =====================================================
# COUNTER
# =====================================================

counter = Counter()


with open(
    ATTR_IMG_FILE,
    "r",
    encoding="utf-8"
) as f:

    lines = f.readlines()[2:]


for line in lines:

    parts = line.split()

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

    for attr in active_attrs:

        if any(
            x in attr
            for x in SILHOUETTE_HINTS
        ):

            counter[
                attr
            ] += 1


# =====================================================
# PRINT
# =====================================================

print(
    "\nSILHOUETTES"
)

print(
    "=" * 60
)

for k, v in (
    counter
    .most_common()
):

    if v >= 5:

        print(
            f"{k:<35} {v}"
        )