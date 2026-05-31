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
# DOWNLOAD IMAGES
# -------------------------

ZIP_URL = (
    "https://github.com/r-vansh/fashion-retrieval-app/releases/download/v1/images.zip"
)

if not os.path.exists(
    "dataset/images"
):

    st.info(
        "Downloading images..."
    )

    zip_path = "images.zip"

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

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as zip_ref:

        zip_ref.extractall(
            "dataset"
        )

    if os.path.exists(
        zip_path
    ):

        os.remove(
            zip_path
        )

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

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 26px !important;
    border: 1px solid #ececec !important;
    background: white !important;
    overflow: hidden !important;
    transition: 0.3s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-4px);
    box-shadow:
    0px 10px 30px rgba(0,0,0,0.08);
}

/* ---------- IMAGE ---------- */

img {
    border-radius: 0px;
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
    metadata_path = (
     "metadata.csv"
    )

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
            download_root="./clip_cache"
        )

        model.eval()

        torch.set_num_threads(1)

        return (
            model,
            preprocess
        )


model, preprocess = load_clip()

# -------------------------
# LOAD EMBEDDINGS
# -------------------------

@st.cache_data
def load_embeddings():

    with open(
        "embeddings.pkl",
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
    query_style="Auto",
    query_silhouette="Auto",
    query_neckline="Auto",
    query_sleeve="Auto",
    query_pattern="Auto",
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

        final_score = (
            similarity * 0.90
        )

        if (
            use_style
            and query_style
            != "Auto"
        ):

            if (
                str(
                    row["style"]
                ).lower()
                ==
                query_style.lower()
            ):

                final_score += 0.04

        if (
            use_silhouette
            and query_silhouette
            != "Auto"
        ):

            if (
                str(
                    row["silhouette"]
                ).lower()
                ==
                query_silhouette.lower()
            ):

                final_score += 0.03

        if (
            use_neckline
            and query_neckline
            != "Auto"
        ):

            if (
                str(
                    row["neckline"]
                ).lower()
                ==
                query_neckline.lower()
            ):

                final_score += 0.015

        if (
            use_sleeve
            and query_sleeve
            != "Auto"
        ):

            if (
                str(
                    row["sleeve"]
                ).lower()
                ==
                query_sleeve.lower()
            ):

                final_score += 0.015

        if (
            use_pattern
            and query_pattern
            != "Auto"
        ):

            if (
                str(
                    row["pattern"]
                ).lower()
                ==
                query_pattern.lower()
            ):

                final_score += 0.015

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

st.sidebar.markdown(
    "<div style='font-size: 32px; font-weight: bold;'>Search Settings</div>",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "Customize how similar references are retrieved."
)

st.sidebar.markdown("---")

with st.sidebar.expander(
    "Attribute Preferences",
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
            )
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
            )
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
            )
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
            )
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
            )
        )
    )

with st.sidebar.expander(
    "Prioritize Attributes",
    expanded=False
):

    use_style = st.checkbox(
        "Style",
        value=False
    )

    use_silhouette = st.checkbox(
        "Silhouette",
        value=False
    )

    use_neckline = st.checkbox(
        "Neckline",
        value=False
    )

    use_sleeve = st.checkbox(
        "Sleeve",
        value=False
    )

    use_pattern = st.checkbox(
        "Pattern",
        value=False
    )

selected_category = (
    st.sidebar.selectbox(
        "Result Category",
        [
            "All",
            "Dress",
            "Top",
            "Skirt",
            "Jacket",
            "Pants"
        ]
    )
)

top_k = st.sidebar.slider(
    "Number of Results",
    3,
    6,
    6
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

        st.markdown("### Query Reference")

        st.image(
            uploaded_image,
            width="stretch"
        )

    with right_col:

        st.markdown(
            f"### Top {top_k} Retrieved References"
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
                selected_category=selected_category,
                query_style=query_style,
                query_silhouette=query_silhouette,
                query_neckline=query_neckline,
                query_sleeve=query_sleeve,
                query_pattern=query_pattern
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
                        image_path
                    ):

                        st.image(
                            image_path,
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

                    st.markdown(
                        f"""
<div style="display:flex; flex-direction:column; gap:4px;">

<div style="
font-size:18px;
font-weight:700;
">
{category_text}
</div>

<div style="display:flex; column-gap:8px; row-gap:8px; flex-wrap:wrap;">

<span style="
background:#F1F1EE;
border: 1px solid #EDEDE9;
padding:6px 12px;
border-radius:20px;
font-size:13px;
">
{style_text}
</span> <span style="
background:#F1F1EE;
border: 1px solid #EDEDE9;
padding:6px 12px;
border-radius:20px;
font-size:13px;
">
{silhouette_text}
</span> <span style="
background:#F1F1EE;
border: 1px solid #EDEDE9;
padding:6px 12px;
border-radius:20px;
font-size:13px;
">
{neckline_text}
</span> <span style="
background:#F1F1EE;
border: 1px solid #EDEDE9;
padding:6px 12px;
border-radius:20px;
font-size:13px;
">
{sleeve_text}
</span> <span style="
background:#F1F1EE;
border: 1px solid #EDEDE9;
padding:6px 12px;
border-radius:20px;
font-size:13px;
">
{pattern_text}
</span>

</div>

<div style="
font-size:14px;
font-weight:500;
color:#555;
">
Visual Match • {match_score}%
</div>

</div>
    """,
    unsafe_allow_html=True
)
                    
