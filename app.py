import streamlit as st
import pandas as pd
import os
import clip
import faiss
import numpy as np
import torch
import pickle
from PIL import Image
import zipfile
import requests

# -------------------------
# FEATURE CONFIGURATION
# -------------------------
SHOW_QUERY_METADATA = True  # Set to False to disable showing metadata of uploaded query images

APP_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------------
# DOWNLOAD IMAGES
# -------------------------

import shutil

DATASET_VERSION = "v2"

ZIP_URL = (
    "https://github.com/r-vansh/fashion-retrieval-app/releases/download/v2/images.zip"
)

DATASET_FOLDER = os.path.join(APP_DIR, "dataset", "images")
VERSION_FILE = os.path.join(APP_DIR, "dataset", "version.txt")
DATASET_PARENT = os.path.join(APP_DIR, "dataset")


def download_dataset():
    # If version file already exists and matches target version (written by another thread), exit early
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, "r") as f:
                installed_version = f.read().strip()
            if installed_version == DATASET_VERSION:
                return
        except Exception:
            pass

    st.info(
        f"Downloading dataset ({DATASET_VERSION})..."
    )

    zip_path = os.path.join(APP_DIR, "images.zip")

    response = requests.get(
        ZIP_URL,
        stream=True
    )

    response.raise_for_status()

    with open(
        zip_path,
        "wb"
    ) as f:

        for chunk in response.iter_content(
            chunk_size=8192
        ):

            f.write(chunk)

    if os.path.exists(zip_path):
        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                DATASET_PARENT
            )

        try:
            os.remove(
                zip_path
            )
        except Exception:
            pass

    with open(
        VERSION_FILE,
        "w"
    ) as f:

        f.write(
            DATASET_VERSION
        )


needs_download = True

if os.path.exists(
    VERSION_FILE
):

    with open(
        VERSION_FILE,
        "r"
    ) as f:

        installed_version = (
            f.read()
            .strip()
        )

    if (
        installed_version
        ==
        DATASET_VERSION
    ):

        needs_download = False


if needs_download:

    if os.path.exists(
        DATASET_FOLDER
    ):

        shutil.rmtree(
            DATASET_FOLDER
        )

    os.makedirs(
        DATASET_PARENT,
        exist_ok=True
    )

    download_dataset()

# Initialize loading state
if "loading_complete" not in st.session_state:
    st.session_state.loading_complete = False

st.markdown("""
<style>

/* ---------- APP ---------- */

.stApp {
    background-color: #f8f8f6;
}

/* ---------- MAIN ---------- */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #ececec;
}

/* ---------- TYPOGRAPHY ---------- */

h1 {
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -2px;
}

h3 {
    font-weight: 700;
}

/* ---------- BUTTONS ---------- */

.stButton button {
    border-radius: 999px;
}

/* ---------- CARD ---------- */

div[data-testid="stVerticalBlockBorderWrapper"],
.stVerticalBlockBorderWrapper {
    border-radius: 4px !important;
    border: 1px solid #ECECEC !important;
    background: white !important;
    padding: 8px !important;
    overflow: visible !important;
    transition: 0.3s ease;
    height: auto !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] div,
.stVerticalBlockBorderWrapper div {
    overflow: visible !important;
    padding: 0px !important;
    margin: 0px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"],
.stVerticalBlockBorderWrapper .stVerticalBlock {
    gap: 8px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover,
.stVerticalBlockBorderWrapper:hover {
    transform: translateY(-4px);
    box-shadow:
    0px 10px 30px rgba(0,0,0,0.08);
}

/* ---------- IMAGE ---------- */

img,
div[data-testid="stVerticalBlockBorderWrapper"] img,
.stVerticalBlockBorderWrapper img {
    border-radius: 2px !important;
}

/* ---------- UPLOAD AREA ---------- */
/* EDIT THESE PROPERTIES: */

[data-testid="stFileUploader"] {
    background-color: #F1F1EE;
    min-height: 180px;
    border-radius: 20px;
    padding: 0;
    color: #000000;
    display: flex;
    flex-direction: column;
    align-items: stretch;
    justify-content: center;
    width: 100%;
}
/* Hide tooltip icon */
[data-testid="stFileUploader"] [class*="tooltipIcon"] {
    display: none !important;
}

[data-testid="stFileUploaderDropzone"] {
    min-height: auto;
    background-color: transparent;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #000000;
    width: 100%;
}

/* Upload button styling */
[data-testid="stFileUploader"] button {
    background-color: #EBEB47;
    color: #000000;
    border-radius: 25px;
    padding: 8px 16px;
    font-size: 14px;
    font-weight: 500;
    border: none;
}

[data-testid="stFileUploader"] button:hover {
    background-color: #E5E51A;
}

/* Upload label text styling */
[data-testid="stFileUploader"] label {
    font-size: 24px;
    font-weight: 600;
    color: #000000;
}

/* Upload info text styling */
[data-testid="stFileUploaderDropzone"] p {
    color: #000000;
    font-size: 14px;
    margin: 0;
    padding: 0;
    line-height: 1;
    height: auto;
}

/* ---------- TAGS ---------- */
.fashion-tag {
    background: #F1F1EE;
    border: 1px solid #EDEDE9;
    padding: 4px 8px;
    border-radius: 20px;
    font-size: 12px;
    display: inline-block;
}

</style>
""", unsafe_allow_html=True)

st.title(
    "MuseMe"
)
st.caption(
    "AI-powered fashion reference retrieval for designers"
)
st.markdown("""
<div style="
padding:24px;
border-radius:20px;
background:#F1F1EE;
border:1px solid #EDEDE9;
margin-bottom:20px;
">

<h3 style="margin: 0; padding: 0;">How It Works</h3>

<ol style="margin-bottom: 0;">
<li>Upload a fashion image or sketch</li>
<li>Customize retrieval settings</li>
<li>Explore visually similar fashion references</li>
</ol>

</div>
""", unsafe_allow_html=True)

#st.write(
#    "Upload a fashion image to find similar references."
#)

# -------------------------
# PATHS
# -------------------------

import glob


# -------------------------
# LOAD DATA WITH SPINNER
# -------------------------

with st.spinner("Loading Fashion Retrieval..."):
    metadata_path = os.path.join(APP_DIR, "metadata.csv")

    metadata = pd.read_csv(
     metadata_path
    )

    # -------------------------
    # LOAD CLIP
    # -------------------------

    device = "cpu"


    @st.cache_resource
    def load_clip():

        model, preprocess = clip.load(
            "ViT-B/32",
            device=device,
            download_root=os.path.join(APP_DIR, "clip_cache")
        )

        model.eval()

        torch.set_num_threads(1)

        return (
            model,
            preprocess
        )


model, preprocess = load_clip()

# -------------------------
# ZERO-SHOT CLASSIFICATION FOR UPLOADED IMAGES
# -------------------------

PROMPT_TEMPLATES = {
    "category": "a photo of a {}",
    "silhouette": "a garment with a {} silhouette",
    "sleeve": "a garment with {}",
    "neckline": "a garment with a {} neckline",
    "color": "a {} colored garment",
    "style": "a {} style outfit",
    "pattern": "a garment with a {} pattern",
}

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

@st.cache_resource
def get_taxonomy_features(_model, _device):
    import json
    with open(os.path.join(APP_DIR, "taxonomy.json"), "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    
    text_features_cache = {}
    for field in ["category", "silhouette", "sleeve", "neckline", "color", "style", "pattern"]:
        options = taxonomy[field]
        prompts = []
        for opt in options:
            key = (field, opt)
            if key in PROMPT_OVERRIDES:
                prompts.append(PROMPT_OVERRIDES[key])
            else:
                template = PROMPT_TEMPLATES.get(field, "a photo of a {}")
                prompts.append(template.format(opt))
        
        tokens = clip.tokenize(prompts).to(_device)
        with torch.no_grad():
            text_feats = _model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        
        text_features_cache[field] = {
            "options": options,
            "features": text_feats
        }
    return text_features_cache

def classify_uploaded_image(uploaded_image, _model, _preprocess, _device):
    image_input = _preprocess(uploaded_image).unsqueeze(0).to(_device)
    with torch.no_grad():
        image_features = _model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    taxonomy_features = get_taxonomy_features(_model, _device)
    
    results = {}
    for field in ["category", "silhouette", "sleeve", "neckline", "color", "style", "pattern"]:
        cache = taxonomy_features[field]
        similarities = (image_features @ cache["features"].T).squeeze(0)
        best_idx = similarities.argmax().item()
        results[field] = cache["options"][best_idx]
    return results


# -------------------------
# LOAD EMBEDDINGS
# -------------------------

@st.cache_data
def load_embeddings():

    with open(
        os.path.join(APP_DIR, "embeddings.pkl"),
        "rb"
    ) as f:

        return pickle.load(
            f
        )


image_paths, image_embeddings = (
    load_embeddings()
)


@st.cache_resource
def load_faiss_index():

    valid_embeddings = []
    original_indices = []

    for i, embedding in enumerate(
        image_embeddings
    ):

        if (
            embedding is None
            or embedding.numel() == 0
        ):

            continue

        vector = (
            embedding
            .cpu()
            .numpy()
            .reshape(-1)
            .astype("float32")
        )

        valid_embeddings.append(
            vector
        )
        original_indices.append(
            i
        )

    if not valid_embeddings:

        raise ValueError(
            "No valid embeddings found."
        )

    vectors = np.stack(
        valid_embeddings
    )
    faiss.normalize_L2(
        vectors
    )

    index = faiss.IndexFlatIP(
        vectors.shape[1]
    )
    index.add(
        vectors
    )

    return (
        index,
        original_indices
    )


faiss_index, faiss_original_indices = (
    load_faiss_index()
)
    
# -------------------------
# FIND SIMILAR
# -------------------------

def find_similar(
    uploaded_image,
    top_k=6,
    use_style=True,
    use_silhouette=True,
    use_neckline=True,
    use_sleeve=True,
    use_pattern=True,
    use_color=True,
    query_style="Auto",
    query_silhouette="Auto",
    query_neckline="Auto",
    query_sleeve="Auto",
    query_pattern="Auto",
    query_color="Auto",
    selected_category="All"
):

    image = preprocess(
        uploaded_image
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        query_embedding = (
            model.encode_image(
                image
            )
        )

        query_embedding /= (
            query_embedding.norm(
                dim=-1,
                keepdim=True
            )
        )

    similarities = []

    query_vector = (
        query_embedding
        .cpu()
        .numpy()
        .astype("float32")
    )
    faiss.normalize_L2(
        query_vector
    )

    faiss_similarities, faiss_positions = (
        faiss_index.search(
            query_vector,
            faiss_index.ntotal
        )
    )

    for similarity, faiss_position in zip(
        faiss_similarities[0],
        faiss_positions[0]
    ):
        i = faiss_original_indices[
            faiss_position
        ]
        similarity = float(
            similarity
        )
        image_path = image_paths[i]

        image_name = (
            os.path.basename(
                image_path
            )
            .replace(".jpg", "")
            .replace(".png", "")
            .strip()
        )

        matched_rows = metadata[
            metadata["image_id"]
            .astype(str)
            .str.strip()
            ==
            image_name
        ]

        if matched_rows.empty:
            continue

        row = matched_rows.iloc[0]
        
        row = row.fillna(
            "Unknown"
        )

        # -------------------------
        # CATEGORY FILTER
        # -------------------------

        if (
            selected_category
            != "All"
        ):

            category = str(
                row.get(
                    "category",
                    ""
                )
            ).strip().lower()

            selected = (
                selected_category
                .strip()
                .lower()
            )

            category = (
                category
                .replace(
                    "pants",
                    "pant"
                )
            )

            selected = (
                selected
                .replace(
                    "pants",
                    "pant"
                )
            )

            if not category:

                continue

            if (
                category
                != selected
            ):

                continue

        # -------------------------
        # HYBRID SCORE
        # -------------------------

        # final_score = (
        #     similarity * 0.90
        # )

        # if (
        #     use_style
        #     and query_style
        #     != "Auto"
        # ):

        #     if (
        #         str(
        #             row["style"]
        #         ).lower()
        #         ==
        #         query_style.lower()
        #     ):

        #         final_score += 0.04

        # if (
        #     use_silhouette
        #     and query_silhouette
        #     != "Auto"
        # ):

        #     if (
        #         str(
        #             row["silhouette"]
        #         ).lower()
        #         ==
        #         query_silhouette.lower()
        #     ):

        #         final_score += 0.03

        # if (
        #     use_neckline
        #     and query_neckline
        #     != "Auto"
        # ):

        #     if (
        #         str(
        #             row["neckline"]
        #         ).lower()
        #         ==
        #         query_neckline.lower()
        #     ):

        #         final_score += 0.015

        # if (
        #     use_sleeve
        #     and query_sleeve
        #     != "Auto"
        # ):

        #     if (
        #         str(
        #             row["sleeve"]
        #         ).lower()
        #         ==
        #         query_sleeve.lower()
        #     ):

        #         final_score += 0.015

        # if (
        #     use_pattern
        #     and query_pattern
        #     != "Auto"
        # ):

        #     if (
        #         str(
        #             row["pattern"]
        #         ).lower()
        #         ==
        #         query_pattern.lower()
        #     ):

        #         final_score += 0.015

        # -------------------------
        # HYBRID SCORE
        # -------------------------

        visual_score = similarity

        metadata_score = 0
        max_metadata_score = 0


        # -------------------------
        # STYLE
        # -------------------------

        if (
            use_style
            and query_style
            != "Auto"
        ):

            weight = 0.25
            max_metadata_score += weight

            if (
                str(row["style"]).lower()
                ==
                query_style.lower()
            ):

                metadata_score += weight


        # -------------------------
        # SILHOUETTE
        # -------------------------

        if (
            use_silhouette
            and query_silhouette
            != "Auto"
        ):

            weight = 0.25
            max_metadata_score += weight

            if (
                str(row["silhouette"]).lower()
                ==
                query_silhouette.lower()
            ):

                metadata_score += weight


        # -------------------------
        # NECKLINE
        # -------------------------

        if (
            use_neckline
            and query_neckline
            != "Auto"
        ):

            weight = 0.35
            max_metadata_score += weight

            if (
                str(row["neckline"]).lower()
                ==
                query_neckline.lower()
            ):

                metadata_score += weight


        # -------------------------
        # SLEEVE
        # -------------------------

        if (
            use_sleeve
            and query_sleeve
            != "Auto"
        ):

            weight = 0.35
            max_metadata_score += weight

            if (
                str(row["sleeve"]).lower()
                ==
                query_sleeve.lower()
            ):

                metadata_score += weight


        # -------------------------
        # PATTERN
        # -------------------------

        if (
            use_pattern
            and query_pattern
            != "Auto"
        ):

            weight = 0.35
            max_metadata_score += weight

            if (
                str(row["pattern"]).lower()
                ==
                query_pattern.lower()
            ):

                metadata_score += weight


        # -------------------------
        # COLOR
        # -------------------------

        if (
            use_color
            and query_color
            != "Auto"
        ):

            weight = 0.5
            max_metadata_score += weight

            if (
                str(row["color"]).lower()
                ==
                query_color.lower()
            ):

                metadata_score += weight


        # -------------------------
        # NORMALIZE METADATA SCORE
        # -------------------------

        if max_metadata_score > 0:

            metadata_score = (
                metadata_score
                / max_metadata_score
            )

        else:

            metadata_score = 0


        # -------------------------
        # FINAL SCORE
        # -------------------------

        final_score = (
            visual_score * 0.75
            +
            metadata_score * 0.25
        )
        
        similarities.append(
            (
                i,
                final_score,
                similarity
            )
        )

    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )
    st.write(
        "VALID RESULTS:",
        len(similarities)
    )
    return similarities[:top_k]

# -------------------------
# UPLOAD IMAGE
# -------------------------

# -------------------------
# SIDEBAR SETTINGS
# -------------------------

def reset_filters():
    st.session_state["filter_category"] = "All"
    st.session_state["filter_top_k"] = 6
    st.session_state["pref_style"] = "Auto"
    st.session_state["pref_silhouette"] = "Auto"
    st.session_state["pref_neckline"] = "Auto"
    st.session_state["pref_sleeve"] = "Auto"
    st.session_state["pref_pattern"] = "Auto"
    st.session_state["pref_color"] = "Auto"
    st.session_state["prioritize_style"] = False
    st.session_state["prioritize_silhouette"] = False
    st.session_state["prioritize_neckline"] = False
    st.session_state["prioritize_sleeve"] = False
    st.session_state["prioritize_pattern"] = False
    st.session_state["prioritize_color"] = False

st.sidebar.markdown(
    "<div style='font-size: 32px; font-weight: bold;'>Search Settings</div>",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "Customize how similar references are retrieved."
)

is_modified = (
    st.session_state.get("filter_category", "All") != "All"
    or st.session_state.get("filter_top_k", 6) != 6
    or st.session_state.get("pref_style", "Auto") != "Auto"
    or st.session_state.get("pref_silhouette", "Auto") != "Auto"
    or st.session_state.get("pref_neckline", "Auto") != "Auto"
    or st.session_state.get("pref_sleeve", "Auto") != "Auto"
    or st.session_state.get("pref_pattern", "Auto") != "Auto"
    or st.session_state.get("pref_color", "Auto") != "Auto"
    or st.session_state.get("prioritize_style", False)
    or st.session_state.get("prioritize_silhouette", False)
    or st.session_state.get("prioritize_neckline", False)
    or st.session_state.get("prioritize_sleeve", False)
    or st.session_state.get("prioritize_pattern", False)
    or st.session_state.get("prioritize_color", False)
)

st.sidebar.button(
    "Reset All Filters",
    on_click=reset_filters,
    disabled=not is_modified,
    use_container_width=True
)

st.sidebar.markdown("---")

with st.sidebar.expander(
    "Tag Preferences",
    expanded=True
):

    style_options = sorted(
        metadata["style"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    query_style = (
        st.selectbox(
            "Style",
            ["Auto"] + list(
                style_options
            ),
            key="pref_style"
        )
    )

    silhouette_options = sorted(
        metadata["silhouette"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    query_silhouette = (
        st.selectbox(
            "Silhouette",
            ["Auto"] + list(
                silhouette_options
            ),
            key="pref_silhouette"
        )
    )

    neckline_options = sorted(
        metadata["neckline"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    query_neckline = (
        st.selectbox(
            "Neckline",
            ["Auto"] + list(
                neckline_options
            ),
            key="pref_neckline"
        )
    )

    sleeve_options = sorted(
        metadata["sleeve"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    query_sleeve = (
        st.selectbox(
            "Sleeve",
            ["Auto"] + list(
                sleeve_options
            ),
            key="pref_sleeve"
        )
    )

    pattern_options = sorted(
        metadata["pattern"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    query_pattern = (
        st.selectbox(
            "Pattern",
            ["Auto"] + list(
                pattern_options
            ),
            key="pref_pattern"
        )
    )

    color_options = sorted(
        metadata["color"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    query_color = (
        st.selectbox(
            "Color",
            ["Auto"] + list(
                color_options
            ),
            key="pref_color"
        )
    )

with st.sidebar.expander(
    "Prioritize Tags",
    expanded=False
):

    use_style = st.checkbox(
        "Style",
        key="prioritize_style"
    )

    use_silhouette = st.checkbox(
        "Silhouette",
        key="prioritize_silhouette"
    )

    use_neckline = st.checkbox(
        "Neckline",
        key="prioritize_neckline"
    )

    use_sleeve = st.checkbox(
        "Sleeve",
        key="prioritize_sleeve"
    )

    use_pattern = st.checkbox(
        "Pattern",
        key="prioritize_pattern"
    )

    use_color = st.checkbox(
        "Color",
        key="prioritize_color"
    )

    category_options = sorted(
        metadata["category"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

selected_category = (
    st.sidebar.selectbox(
        "Result Category",
            ["All"] + [
                category.title()
                for category in category_options
            ],
            key="filter_category"
    )
)

top_k = st.sidebar.slider(
    "Number of Results",
    3,
    6,
    6,
    key="filter_top_k"
)


# -------------------------
# IMAGE UPLOAD
# -------------------------


st.markdown(f"""
<style>
[data-testid="stFileUploader"] button {{
    display: flex;
    align-items: center;
    gap: 8px;
}}
[data-testid="stFileUploader"] button > span {{
    display: none;
}}
[data-testid="stFileUploader"] button::after {{
    content: "";
    font-size: 14px;
    color: #000000;
}}
/* Hide all buttons except first one */
[data-testid="stFileUploader"] button:nth-child(n+2) {{
    display: none !important;
}}
/* Hide clear button */
[data-testid="stFileUploader"] [role="button"] {{
    display: none !important;
}}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        label="Upload fashion images",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
        
    )

with col2:
    st.write("")

# -------------------------
# RETRIEVAL
# -------------------------

# if not uploaded_file:

#    st.markdown("""
#        <br><br>

#         ## Upload a Fashion Reference

#        Use an inspiration image, fashion photograph, garment detail, or sketch to retrieve visually similar references.

#        """, unsafe_allow_html=True)

if uploaded_file:

    uploaded_image = Image.open(
        uploaded_file
    ).convert("RGB")

    uploaded_image.thumbnail(
        (512, 512)
    )

    st.markdown("---")

    left_col, right_col = st.columns(
        [1.2, 2.8]
    )

    with left_col:

        st.markdown("### Reference")

        st.image(
            uploaded_image,
            width="stretch"
        )

        if SHOW_QUERY_METADATA:
            with st.expander("Query Metadata", expanded=False):
                # Check metadata.csv first (just in case)
                query_image_name = (
                    os.path.basename(uploaded_file.name)
                    .replace(".jpg", "")
                    .replace(".png", "")
                    .replace(".jpeg", "")
                    .strip()
                )
                query_matched_rows = metadata[
                    metadata["image_id"]
                    .astype(str)
                    .str.strip()
                    ==
                    query_image_name
                ]

                if not query_matched_rows.empty:
                    query_row = query_matched_rows.iloc[0]
                    query_row = query_row.fillna("Unknown")
                    q_category = str(query_row.get("category", "Unknown")).title()
                    q_style = str(query_row.get("style", "Unknown")).title()
                    q_silhouette = str(query_row.get("silhouette", "Unknown")).title()
                    q_neckline = str(query_row.get("neckline", "Unknown")).title()
                    q_sleeve = str(query_row.get("sleeve", "Unknown")).title()
                    q_pattern = str(query_row.get("pattern", "Unknown")).title()
                    q_color = str(query_row.get("color", "Unknown")).title()
                else:
                    # Classify the uploaded image dynamically
                    predicted_meta = classify_uploaded_image(uploaded_image, model, preprocess, device)
                    q_category = str(predicted_meta.get("category", "Unknown")).title()
                    q_style = str(predicted_meta.get("style", "Unknown")).title()
                    q_silhouette = str(predicted_meta.get("silhouette", "Unknown")).title()
                    q_neckline = str(predicted_meta.get("neckline", "Unknown")).title()
                    q_sleeve = str(predicted_meta.get("sleeve", "Unknown")).title()
                    q_pattern = str(predicted_meta.get("pattern", "Unknown")).title()
                    q_color = str(predicted_meta.get("color", "Unknown")).title()

                st.markdown(
                    f"""
<div style="display:flex; flex-direction:column; gap:8px;">
    <div style="font-size:18px; font-weight:700;">{q_category}</div>
    <div style="display:flex; flex-wrap:wrap; gap:8px;">
        <span class="fashion-tag">{q_style}</span>
        <span class="fashion-tag">{q_silhouette}</span>
        <span class="fashion-tag">{q_neckline}</span>
        <span class="fashion-tag">{q_sleeve}</span>
        <span class="fashion-tag">{q_pattern}</span>
        <span class="fashion-tag">{q_color}</span>
    </div>
</div>
                    """,
                    unsafe_allow_html=True
                )



    with right_col:

        st.markdown(
            f"### Top {top_k} Retrieved Results"
        )

        with st.status("Finding Similar Designs...", expanded=True) as status:

            results = find_similar(
                uploaded_image,
                top_k=top_k,
                use_style=use_style,
                use_silhouette=use_silhouette,
                use_neckline=use_neckline,
                use_sleeve=use_sleeve,
                use_pattern=use_pattern,
                use_color=use_color,
                selected_category=selected_category,
                query_style=query_style,
                query_silhouette=query_silhouette,
                query_neckline=query_neckline,
                query_sleeve=query_sleeve,
                query_pattern=query_pattern,
                query_color=query_color
            )
            
            status.update(label="Designs Found!", state="complete")
            if len(results) == 0:

                st.warning(
                    "No matching results found. Try changing category or metadata filters."
                )

                st.stop()
        cols = st.columns(
            3,
            gap="small"
        )

        for i, (
            idx,
            score,
            visual_similarity
        ) in enumerate(results):

            image_path = image_paths[
                idx
            ]

            resolved_image_path = image_path
            if not os.path.exists(resolved_image_path):
                resolved_image_path = os.path.join(APP_DIR, image_path)

            image_name = (
                os.path.basename(
                    image_path
                )
                .replace(".jpg", "")
                .replace(".png", "")
                .strip()
            )

            matched_rows = metadata[
                metadata["image_id"]
                .astype(str)
                .str.strip()
                ==
                image_name.strip()
            ]

            if matched_rows.empty:
                continue

            row = matched_rows.iloc[0]

            with cols[i % 3]:

                card = st.container(
                    border=True
                )

                with card:

                    if os.path.exists(
                        resolved_image_path
                    ):

                        st.image(
                            resolved_image_path,
                            width="stretch"
                        )

                    else:

                        continue

                    match_score = int(
                        visual_similarity * 100
                    )

                    category_text = str(
                        row.get(
                            "category",
                            "Unknown"
                        )
                    ).title()

                    style_text = str(
                        row.get(
                            "style",
                            "Unknown"
                        )
                    ).title()

                    silhouette_text = str(
                        row.get(
                            "silhouette",
                            "Unknown"
                        )
                    ).title()

                    neckline_text = str(
                        row.get(
                            "neckline",
                            "Unknown"
                        )
                    ).title()

                    sleeve_text = str(
                        row.get(
                            "sleeve",
                            "Unknown"
                        )
                    ).title()

                    pattern_text = str(
                        row.get(
                            "pattern",
                            "Unknown"
                        )
                    ).title()

                    color_text = str(
                        row.get(
                            "color",
                            "Unknown"
                        )
                    ).title()

                    st.markdown(
                        f"""
<div style="display:flex; flex-direction:column; gap:8px;">
    <div style="display:flex; justify-content:space-between; align-items:baseline; width:100%;">
        <span style="font-size:18px; font-weight:700;">{category_text}</span>
        <span style="font-size:14px; font-weight:500; color:#555;">Visual Match • {match_score}%</span>
    </div>
    <div style="display:flex; flex-wrap:wrap; gap:8px;">
        <span class="fashion-tag">{style_text}</span>
        <span class="fashion-tag">{silhouette_text}</span>
        <span class="fashion-tag">{neckline_text}</span>
        <span class="fashion-tag">{sleeve_text}</span>
        <span class="fashion-tag">{pattern_text}</span>
        <span class="fashion-tag">{color_text}</span>
    </div>
</div>
    """,
    unsafe_allow_html=True
)
                    
