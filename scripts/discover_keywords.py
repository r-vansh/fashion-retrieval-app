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
# CANDIDATE KEYWORDS
# =====================================================

SLEEVE_HINTS = [
    "sleeve",
    "sleeveless",
    "batwing",
    "raglan",
    "dolman",
    "flutter",
    "cap-sleeve",
    "bell-sleeve",
    "cuffed"
]

NECKLINE_HINTS = [
    "neck",
    "collar",
    "halter",
    "strapless",
    "sweetheart",
    "square",
    "asymmetrical"
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
    "embroidered"
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
    "tailored"
]


# =====================================================
# COUNTERS
# =====================================================

sleeve_counter = Counter()
neckline_counter = Counter()
pattern_counter = Counter()
style_counter = Counter()


# =====================================================
# PROCESS ATTRIBUTES
# =====================================================

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

        # Sleeve
        if any(
            x in attr
            for x in SLEEVE_HINTS
        ):
            sleeve_counter[
                attr
            ] += 1

        # Neckline
        if any(
            x in attr
            for x in NECKLINE_HINTS
        ):
            neckline_counter[
                attr
            ] += 1

        # Pattern
        if any(
            x in attr
            for x in PATTERN_HINTS
        ):
            pattern_counter[
                attr
            ] += 1

        # Style
        if any(
            x in attr
            for x in STYLE_HINTS
        ):
            style_counter[
                attr
            ] += 1


# =====================================================
# PRINT
# =====================================================

MIN_FREQ = 5


def print_valid(
    title,
    counter
):
    print(
        "\n"
        + "=" * 60
    )

    print(
        title.upper()
    )

    print(
        "=" * 60
    )

    for k, v in (
        counter
        .most_common()
    ):

        if v >= MIN_FREQ:

            print(
                f"{k:<35} {v}"
            )


print_valid(
    "Sleeve",
    sleeve_counter
)

print_valid(
    "Neckline",
    neckline_counter
)

print_valid(
    "Pattern",
    pattern_counter
)

print_valid(
    "Style",
    style_counter
)