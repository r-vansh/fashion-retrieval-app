import csv
from io import StringIO
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "metadata_v2.csv"

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


def main():
    metadata = pd.read_csv(INPUT_PATH)
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
            f"Reviewed image IDs do not match metadata.csv. "
            f"Missing: {sorted(missing_ids)}. Extra: {sorted(extra_ids)}."
        )

    columns_to_refine = [
        "category",
        "silhouette",
        "sleeve",
        "neckline",
        "style",
    ]

    metadata = metadata.drop(
        columns=columns_to_refine
    ).merge(
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

    metadata.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved {len(metadata)} reviewed rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
