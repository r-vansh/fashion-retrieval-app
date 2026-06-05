import os
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# -------------------------
# CONFIG
# -------------------------

IMAGES_DIR = "dataset/images"
METADATA_PATH = "metadata.csv"
TAXONOMY_PATH = "taxonomy.json"

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

TAG_FIELDS = [
    "category",
    "silhouette",
    "sleeve",
    "neckline",
    "color",
    "style",
    "pattern",
]

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="MuseMe Metadata Reviewer",
    layout="wide",
)

st.markdown("""
<h1 style="
    font-size: 1.9rem;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
">
MuseMe Metadata Reviewer
</h1>
""", unsafe_allow_html=True)

if st.session_state.get("show_save_success", False):
    st.toast("💾 Metadata saved successfully!")
    st.session_state.show_save_success = False

# -------------------------
# LOAD TAXONOMY
# -------------------------

@st.cache_data
def load_taxonomy():
    with open(
        TAXONOMY_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


taxonomy = load_taxonomy()

# add blank option
for key in taxonomy:
    taxonomy[key] = [""] + taxonomy[key]


def get_selectbox_options(field, existing_val):
    options = list(taxonomy.get(field, []))
    if existing_val and existing_val not in options:
        options.append(existing_val)
    return options


# -------------------------
# LOAD METADATA
# -------------------------

def load_metadata():

    if os.path.exists(
        METADATA_PATH
    ):

        df = pd.read_csv(
            METADATA_PATH
        )

        # ensure all columns exist
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        return df

    return pd.DataFrame(
        columns=CSV_COLUMNS
    )


metadata_df = load_metadata()

# Create a fast O(1) dictionary metadata lookup
metadata_lookup = {row["file_name"]: row for _, row in metadata_df.iterrows()}


# -------------------------
# COMPLETENESS & COUNT HELPERS
# -------------------------

def count_filled_fields(row):

    filled = 0

    for field in TAG_FIELDS:

        value = row.get(
            field,
            ""
        )

        # ignore NaN properly
        if pd.isna(value):
            continue

        value = str(
            value
        ).strip().lower()

        # empty or literal nan or none = unfilled (as requested by user)
        if value in [
            "",
            "nan",
            "none"
        ]:
            continue

        filled += 1

    return filled


def count_missing_fields(row):

    missing = 0

    for field in TAG_FIELDS:

        value = row.get(
            field,
            ""
        )

        if pd.isna(value):
            missing += 1
            continue

        value = str(
            value
        ).strip().lower()

        if value in [
            "",
            "nan",
            "none"
        ]:
            missing += 1

    return missing


def get_missing_count_for_file(file_name):

    row = metadata_lookup.get(file_name)

    if row is None:
        return len(TAG_FIELDS)

    return count_missing_fields(row)


# -------------------------
# LOAD & INITIALIZE IMAGES (ONCE PER SESSION)
# -------------------------

if "image_files" not in st.session_state:

    raw_files = sorted([
        f
        for f in os.listdir(
            IMAGES_DIR
        )
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            )
        )
    ])

    if not raw_files:
        st.error(
            "No images found in dataset/images/"
        )
        st.stop()

    # Sort files by completeness: most missing fields (highest missing count) first.
    # We use the lookup dictionary to do this in O(N) overall instead of O(N^2).
    sorted_files = sorted(
        raw_files,
        key=lambda name: (
            -get_missing_count_for_file(name),
            name
        )
    )

    st.session_state.image_files = sorted_files

image_files = st.session_state.image_files


# -------------------------
# FIND FIRST INCOMPLETE
# -------------------------

def find_first_incomplete():

    for i, img in enumerate(
        image_files
    ):

        row = metadata_lookup.get(img)

        # no metadata row
        if row is None:
            return i

        filled_count = count_filled_fields(row)

        # All 7 tag fields must be filled (non-none) to be complete
        if filled_count < len(TAG_FIELDS):
            return i

    return 0


# -------------------------
# SESSION STATE
# -------------------------

# initialize once
if (
    "initialized"
    not in st.session_state
):

    st.session_state.current_index = (
        find_first_incomplete()
    )

    st.session_state.initialized = True


# -------------------------
# CURRENT IMAGE
# -------------------------

current_file = image_files[
    st.session_state.current_index
]

current_path = os.path.join(
    IMAGES_DIR,
    current_file
)

# -------------------------
# EXISTING METADATA
# -------------------------

existing_row = metadata_lookup.get(current_file)
    
# -------------------------
# LAYOUT
# -------------------------

# -------------------------
# COMPACT SINGLE SCREEN UI
# -------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 0.2rem !important;
    padding-bottom: 0rem !important;
    max-width: 100% !important;
}

header {
    visibility: hidden;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.35rem !important;
}

.stSelectbox label,
.stTextArea label {
    font-size: 0.9rem !important;
}

button {
    height: 42px !important;
}

/* make image fit better */
[data-testid="stImage"] img {
    max-height: 420px !important;
    object-fit: contain !important;
}
</style>
""", unsafe_allow_html=True)


top_left, top_right = st.columns(
    [1.5, 1],
    vertical_alignment="top"
)

# -------------------------
# LEFT SIDE IMAGE
# -------------------------

with top_left:

    components.html(
        f"""
<div style="
    display:flex;
    align-items:center;
    gap:12px;
    margin-bottom:6px;
    width:100%;
">

    <span style="
        font-size:1.5rem;
        font-weight:700;
        flex-shrink:0;
    ">
        {st.session_state.current_index + 1} / {len(image_files)}
    </span>

    <span style="
        font-size:0.95rem;
        color:#777;
        overflow:hidden;
        text-overflow:ellipsis;
        white-space:nowrap;
        flex:1;
    ">
        {current_file}
    </span>

</div>
""",
        height=38
    )

    image = Image.open(
        current_path
    )

    image_container = st.container(
        height=420
    )

    with image_container:

        st.image(
            image,
            use_container_width=True
        )

# -------------------------
# RIGHT SIDE CONTROLS
# -------------------------

with top_right:

    st.markdown("""
    <h3 style="
        margin:0 0 0.5rem 0;
        font-size:1.5rem;
    ">
    Metadata
    </h3>
    """, unsafe_allow_html=True)

    def get_existing_value(
        field
    ):
        if (
            existing_row
            is not None
        ):
            value = existing_row.get(
                field,
                ""
            )

            if pd.isna(value):
                return ""

            return str(value)

        return ""

    jump_col1, jump_col2 = st.columns([2, 1])

    with jump_col1:

        jump_to = st.number_input(
            "Jump",
            min_value=1,
            max_value=len(image_files),
            value=int(
                st.session_state.current_index + 1
            )
        )

    with jump_col2:

        st.write("")
        st.write("")

        if st.button(
            "Go",
            use_container_width=True
        ):

            st.session_state.current_index = (
                jump_to - 1
            )

            st.rerun()

    with st.form(
        "metadata_form",
        clear_on_submit=False
    ):

        category_opts = get_selectbox_options("category", get_existing_value("category"))
        category = st.selectbox(
            "Category",
            category_opts,
            index=category_opts.index(get_existing_value("category")) if get_existing_value("category") else 0
        )

        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            sleeve_opts = get_selectbox_options("sleeve", get_existing_value("sleeve"))
            sleeve = st.selectbox(
                "Sleeve",
                sleeve_opts,
                index=sleeve_opts.index(get_existing_value("sleeve")) if get_existing_value("sleeve") else 0
            )

        with row1_col2:
            neckline_opts = get_selectbox_options("neckline", get_existing_value("neckline"))
            neckline = st.selectbox(
                "Neckline",
                neckline_opts,
                index=neckline_opts.index(get_existing_value("neckline")) if get_existing_value("neckline") else 0
            )

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            color_opts = get_selectbox_options("color", get_existing_value("color"))
            color = st.selectbox(
                "Color",
                color_opts,
                index=color_opts.index(get_existing_value("color")) if get_existing_value("color") else 0
            )

        with row2_col2:
            pattern_opts = get_selectbox_options("pattern", get_existing_value("pattern"))
            pattern = st.selectbox(
                "Pattern",
                pattern_opts,
                index=pattern_opts.index(get_existing_value("pattern")) if get_existing_value("pattern") else 0
            )

        row3_col1, row3_col2 = st.columns(2)

        with row3_col1:
            silhouette_opts = get_selectbox_options("silhouette", get_existing_value("silhouette"))
            silhouette = st.selectbox(
                "Silhouette",
                silhouette_opts,
                index=silhouette_opts.index(get_existing_value("silhouette")) if get_existing_value("silhouette") else 0
            )

        with row3_col2:
            style_opts = get_selectbox_options("style", get_existing_value("style"))
            style = st.selectbox(
                "Style",
                style_opts,
                index=style_opts.index(get_existing_value("style")) if get_existing_value("style") else 0
            )

        extra_notes = st.text_area(
            "Notes",
            value=get_existing_value(
                "extra_notes"
            ),
            height=70
        )

        form_col1, form_col2 = st.columns(2)

        with form_col1:
            save_clicked = st.form_submit_button(
                "💾 Save"
            )

        with form_col2:
            save_next_clicked = st.form_submit_button(
                "✅ Save & Next"
            )
    
# -------------------------
# SAVE FUNCTION
# -------------------------

def save_metadata():

    global metadata_df

    existing_index = metadata_df[
        metadata_df[
            "file_name"
        ] == current_file
    ].index

    # preserve image_id or generate it if blank
    image_id = ""

    if len(existing_index) > 0:
        image_id = metadata_df.loc[
            existing_index[0],
            "image_id"
        ]
        if pd.isna(image_id):
            image_id = ""

    if not image_id:
        image_id = os.path.splitext(current_file)[0]

    row_data = {
        "image_id":
            image_id,

        "file_name":
            current_file,

        "category":
            category,

        "silhouette":
            silhouette,

        "sleeve":
            sleeve,

        "neckline":
            neckline,

        "color":
            color,

        "style":
            style,

        "pattern":
            pattern,

        "extra_notes":
            extra_notes,
    }

    # update existing
    if len(existing_index) > 0:

        for key, value in row_data.items():

            metadata_df.loc[
                existing_index[0],
                key
            ] = value

    # create new
    else:

        metadata_df = pd.concat(
            [
                metadata_df,
                pd.DataFrame(
                    [row_data]
                )
            ],
            ignore_index=True
        )

    try:
        metadata_df.to_csv(
            METADATA_PATH,
            index=False
        )
    except Exception as e:
        st.error(f"Error writing to metadata.csv: {e}. If the file is open in another program (like Excel), please close it and try again.")
        st.stop()


# -------------------------
# BUTTONS
# -------------------------

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:

    if st.button(
        "⬅ Previous",
        use_container_width=True
    ):

        if (
            st.session_state.current_index
            > 0
        ):

            st.session_state.current_index -= 1
            st.rerun()


with col2:

    if st.button(
        "⏭ Skip",
        use_container_width=True
    ):

        if (
            st.session_state.current_index
            <
            len(image_files) - 1
        ):

            st.session_state.current_index += 1
            st.rerun()


with col3:

    st.write("")


with col4:

    st.write("")


if save_clicked or save_next_clicked:

    save_metadata()

    if save_next_clicked and (
        st.session_state.current_index
        <
        len(image_files) - 1
    ):

        st.session_state.current_index += 1

    st.session_state.show_save_success = True
    st.rerun()


# -------------------------
# QUICK STATUS
# -------------------------

completed = sum(
    1
    for img in image_files
    if img in metadata_lookup
    and count_filled_fields(metadata_lookup[img]) >= len(TAG_FIELDS)
)

progress = (
    completed
    /
    len(image_files)
)

st.progress(
    progress
)

st.caption(
    f"{completed} / {len(image_files)} completed"
)

if st.button(
    "⏩ Jump to First Incomplete"
):

    next_index = find_first_incomplete()

    st.session_state.current_index = (
        next_index
    )

    st.rerun()