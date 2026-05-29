import streamlit as st
import pandas as pd
import os
import clip
import torch
import pickle
from PIL import Image
import zipfile
import requests


# -------------------------
# DOWNLOAD IMAGES
# -------------------------

ZIP_URL = (
    "https://github.com/r-vansh/fashion-retrieval-app/releases/download/v1/images.zip?raw=1"
)

if not os.path.exists(
    "dataset/images"
):

    st.info(
        "Downloading images..."
    )

    zip_path = (
        "images.zip"
    )

    response = requests.get(
        ZIP_URL,
        allow_redirects=True
    )
    response.raise_for_status()

    with open(
        zip_path,
        "wb"
    ) as f:

        f.write(
            response.content
        )

    st.write(
        os.path.getsize(
            zip_path
        )
    )

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

st.set_page_config(
    page_title="Fashion Retrieval",
    layout="wide"
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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, preprocess = clip.load(
        "ViT-B/32",
        device=device,
        download_root="./clip_cache"
    )

    # -------------------------
    # LOAD EMBEDDINGS
    # -------------------------

    with open(
        "embeddings.pkl",
        "rb"
    ) as f:

        loaded_data = pickle.load(f)
        

# -------------------------
# LOAD EMBEDDINGS
# -------------------------

image_embeddings = (
    loaded_data[0]
)

image_paths = (
    loaded_data[1]
)

# -------------------------
# CONVERT TO TENSORS
# -------------------------

fixed_embeddings = []

for emb in image_embeddings:

    try:

        # already tensor
        if isinstance(
            emb,
            torch.Tensor
        ):

            fixed_embeddings.append(
                emb.cpu()
            )

        else:

            fixed_embeddings.append(
                torch.as_tensor(
                    emb
                ).cpu()
            )

    except Exception:

        # skip broken embeddings
        continue


image_embeddings = (
    fixed_embeddings
)

# FIX IMAGE PATHS
# -------------------------

fixed_paths = []

for path in image_paths:

    if path is None:

        continue

    path = str(path)

    filename = os.path.basename(
        path
    )

    fixed_path = os.path.join(
        "dataset",
        "images",
        filename
    )

    fixed_paths.append(
        fixed_path
    )

image_paths = fixed_paths
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

    similarities = []

    for i, emb in enumerate(
        image_embeddings
    ):

        similarity = (
            torch.cosine_similarity(
                query_embedding.cpu(),
                emb
            ).item()
        )

        image_path = image_paths[i]

        image_name = os.path.basename(
            image_path
        ).replace(".jpg", "")

        row = metadata[
            metadata["image_id"]
            == image_name
        ].iloc[0]

        # -------------------------
        # CATEGORY FILTER
        # -------------------------

        category = str(
            row["category"]
        ).strip().lower()

        selected = str(
            selected_category
        ).strip().lower()

        # normalize pants/pant
        if category == "pants":
            category = "pant"

        if selected == "pants":
            selected = "pant"

        if selected != "all":

            if category != selected:
                continue

        # -------------------------
        # SCORING
        # -------------------------

        final_score = similarity

        if use_style:

            style = str(
                row["style"]
            ).lower()

            if style != "na":
                final_score += 0.20

        if use_silhouette:

            silhouette = str(
                row["silhouette"]
            ).lower()

            if silhouette != "na":
                final_score += 0.30
        
        if use_neckline:

            neckline = str(
                row["neckline"]
            ).lower()

            if neckline != "none":
                final_score += 0.15        

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

    # return similarities[:top_k]
    candidate_pool = similarities[:80]
    reranked = []

    for idx, score, visual_similarity in candidate_pool:

        image_path = image_paths[idx]

        image_name = os.path.basename(
            image_path
        ).replace(".jpg", "")

        row = metadata[
            metadata["image_id"]
            == image_name
        ].iloc[0]

        rerank_score = score

        # -------------------------
        # QUERY METADATA MATCH
        # -------------------------

        if (
            use_style
            and query_style != "Auto"
        ):

            if (
                str(row["style"])
                .lower()
                ==
                query_style.lower()
            ):
                
                rerank_score += 0.12


        if (
            use_silhouette
            and query_silhouette
            != "Auto"
        ):

            if (
                str(row["silhouette"])
                .lower()
                ==
                query_silhouette.lower()
            ):
                
                rerank_score += 0.18


        if (
            use_neckline
            and query_neckline
            != "Auto"
        ):

            if (
                str(row["neckline"])
                .lower()
                ==
                query_neckline.lower()
            ):
                
                rerank_score += 0.10
                
        if (
            use_sleeve
            and query_sleeve
            != "Auto"
        ):

            if (
                str(row["sleeve"]).lower()
                ==
                query_sleeve.lower()
            ):

                rerank_score += 0.12
                
        if (
            use_pattern
            and query_pattern
            != "Auto"
        ):

            if (
                str(row["pattern"]).lower()
                ==
                query_pattern.lower()
            ):

                rerank_score += 0.10    


        reranked.append(
            (
                idx,
                rerank_score,
                visual_similarity
            )
        )

    reranked.sort(
        key=lambda x: x[1],
        reverse=True
    )
    
    return reranked[:top_k]

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

st.sidebar.subheader(
    "Query Metadata"
)

style_options = sorted(
    metadata["style"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .unique()
)

query_style = (
    st.sidebar.selectbox(
        "Style",
        ["Auto"] + list(style_options)
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
    st.sidebar.selectbox(
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
    st.sidebar.selectbox(
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
    st.sidebar.selectbox(
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
    st.sidebar.selectbox(
        "Pattern",
        ["Auto"] + list(
            pattern_options
        )
    )
)

use_style = st.sidebar.checkbox(
    "Style Match",
    value=True
)

use_silhouette = (
    st.sidebar.checkbox(
        "Silhouette Match",
        value=True
    )
)

use_neckline = (
    st.sidebar.checkbox(
        "Neckline Match",
        value=True
    )
)

use_sleeve = (
    st.sidebar.checkbox(
        "Sleeve Match",
        value=True
    )
)

use_pattern = (
    st.sidebar.checkbox(
        "Pattern Match",
        value=True
    )
)

top_k = st.sidebar.slider(
    "Number of Results",
    3,
    10,
    6
)

selected_category = (
    st.sidebar.selectbox(
        "Category",
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
            
            st.write(
                "RESULTS:",
                len(results)
            )

            st.write(
                results[:2]
            )
            
            status.update(label="Designs Found!", state="complete")

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

            image_name = os.path.basename(
                image_path
            ).replace(".jpg","")

            row = metadata[
                metadata["image_id"]
                == image_name
            ].iloc[0]

            with cols[i % 3]:

                card = st.container(
                    border=True
                )

                with card:

                    st.write(
                        "PATH:",
                        image_path
                    )

                    exists = os.path.exists(
                        image_path
                    )

                    st.write(
                        "EXISTS:",
                        exists
                    )

                    if exists:

                        try:

                            st.write(
                                image_path
                            )

                            st.write(
                                os.path.exists(
                                    image_path
                                )
                            )

                            if os.path.exists(
                                image_path
                            ):

                                st.image(
                                    image_path,
                                    width="stretch"
                                )

                            else:

                                st.error(
                                    f"Missing: {image_path}"
                                )

                        except Exception as e:

                            st.error(
                                str(e)
                            )

                    else:

                        st.error(
                            "Image missing"
                        )

                    match_score = min(
                        max(
                            int(score * 100),
                            50
                        ),
                        98
                    )
                    
                    

                    match_score = int(
                        visual_similarity * 100
                    )

                    st.markdown(
                        f"""
<div style="display:flex; flex-direction:column; gap:4px;">

<div style="
font-size:18px;
font-weight:700;
">
{row['category'].title()}
</div>

<div style="display:flex; column-gap:8px; row-gap:8px; flex-wrap:wrap;">

<span style="
background:#F1F1EE;
border: 1px solid #EDEDE9;
padding:6px 12px;
border-radius:20px;
font-size:13px;
">
{row['style'].title()}
</span> <span style="
background:#F1F1EE;
border: 1px solid #EDEDE9;
padding:6px 12px;
border-radius:20px;
font-size:13px;
">
{row['silhouette'].title()}
</span> <span style="
background:#F1F1EE;
border: 1px solid #EDEDE9;
padding:6px 12px;
border-radius:20px;
font-size:13px;
">
{row['neckline'].title()}
</span> <span style="
background:#F1F1EE;
border: 1px solid #EDEDE9;
padding:6px 12px;
border-radius:20px;
font-size:13px;
">
{row['sleeve'].title()}
</span> <span style="
background:#F1F1EE;
border: 1px solid #EDEDE9;
padding:6px 12px;
border-radius:20px;
font-size:13px;
">
{row['pattern'].title()}
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
                    