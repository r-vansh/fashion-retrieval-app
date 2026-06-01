import os
import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OLD_CSV = os.path.join(
    PROJECT_ROOT,
    "metadata.csv"
)

NEW_CSV = os.path.join(
    PROJECT_ROOT,
    "deepfashion_metadata_enriched.csv"
)

OUTPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "metadata.csv"
)


old_df = pd.read_csv(
    OLD_CSV
)

new_df = pd.read_csv(
    NEW_CSV
)

print(
    f"Old rows: {len(old_df)}"
)

print(
    f"New rows: {len(new_df)}"
)


merged_df = pd.concat(
    [
        old_df,
        new_df
    ],
    ignore_index=True
)


merged_df = (
    merged_df
    .drop_duplicates(
        subset="image_id"
    )
)


merged_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nDONE")

print(
    f"Final rows: "
    f"{len(merged_df)}"
)

print(
    "\nCategory counts:"
)

print(
    merged_df[
        "category"
    ].value_counts()
)