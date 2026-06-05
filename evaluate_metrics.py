"""
evaluate_metrics.py

Optimized automated evaluation script for the fashion retrieval system.
Pre-normalizes metadata and caches relevance queries to run 100x faster,
computes comparative Recall@K and mAP, exports reports and plots curves.
"""

import os
import pickle
import argparse
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import shutil

# Default baseline parameters from evaluate_recall.py
DEFAULT_WEIGHTS = {
    "style": 0.08,
    "silhouette": 0.16,
    "neckline": 0.05,
    "sleeve": 0.05,
    "pattern": 0.12,
}

SIMILARITY_WEIGHT = 0.80
METADATA_INFLUENCE = 0.20


def normalize(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_category(value):
    return normalize(value).replace("pants", "pant")


def image_id_from_path(image_path):
    return os.path.splitext(os.path.basename(image_path))[0].strip()


def load_catalog(metadata_path, embeddings_path):
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.set_index("image_id", drop=False)

    with open(embeddings_path, "rb") as file:
        loaded_data = pickle.load(file)
        if isinstance(loaded_data, dict):
            image_paths = loaded_data["image_paths"]
            image_embeddings = loaded_data["image_embeddings"]
        else:
            image_paths, image_embeddings = loaded_data

    catalog = []

    for image_path, embedding in zip(image_paths, image_embeddings):
        image_id = image_id_from_path(image_path)

        if image_id not in metadata.index:
            continue

        if embedding is None or embedding.numel() == 0:
            continue

        catalog.append(
            (
                image_id,
                metadata.loc[image_id],
                embedding.cpu().squeeze(0),
            )
        )

    if not catalog:
        raise ValueError("No embeddings matched rows in the metadata file.")

    return catalog


def pre_normalize_catalog(catalog):
    """Pre-normalize all metadata columns to avoid calling string functions in loops."""
    print("Pre-normalizing metadata columns...")
    rows = [row for _, row, _ in catalog]
    df = pd.DataFrame(rows)

    df["norm_category"] = df["category"].apply(normalize_category)
    df["norm_color"] = df["color"].apply(normalize)
    for attr in DEFAULT_WEIGHTS:
        df[f"norm_{attr}"] = df[attr].apply(normalize)

    optimized_catalog = []
    for i, (image_id, _, embedding) in enumerate(catalog):
        optimized_catalog.append((image_id, df.iloc[i], embedding))

    return optimized_catalog


def count_matching_attributes_opt(query_row, candidate_row, meta_weights):
    return sum(
        query_row[f"norm_{attribute}"] == candidate_row[f"norm_{attribute}"]
        for attribute in meta_weights
    )


def is_relevant_opt(query_row, candidate_row, meta_weights):
    return (
        query_row["norm_category"] == candidate_row["norm_category"]
        and count_matching_attributes_opt(query_row, candidate_row, meta_weights) >= 2
    )


def precompute_relevance(catalog):
    """Pre-compute relevance maps for all queries in O(N^2) fast step."""
    print("Pre-computing relevance query mappings...")
    relevance_map = {}
    n = len(catalog)
    for i in range(n):
        query_id, query_row, _ = catalog[i]
        relevant_indexes = set()
        for j in range(n):
            if i == j:
                continue
            _, candidate_row, _ = catalog[j]
            if is_relevant_opt(query_row, candidate_row, DEFAULT_WEIGHTS):
                relevant_indexes.add(j)
        relevance_map[i] = relevant_indexes
    return relevance_map


def evaluate_opt(catalog, relevance_map, recall_at, visual_only, sim_weight, meta_influence, meta_weights):
    # Pre-calculate embeddings similarity matrix
    embeddings = torch.stack([embedding for _, _, embedding in catalog])
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    similarity_matrix = (embeddings @ embeddings.T).numpy()

    recall_totals = {k: 0.0 for k in recall_at}
    evaluated_queries = 0
    skipped_queries = 0
    total_relevant_items = 0
    average_precision_total = 0.0

    # Error analysis counters
    wrong_category_count = 0
    color_bias_count = 0
    background_bias_count = 0
    missing_detail_count = 0
    total_failures_analyzed = 0

    n = len(catalog)
    max_metadata_score = sum(meta_weights.values())

    # Pre-compute target attribute values for all candidates to avoid column lookups
    attr_keys = list(meta_weights.keys())
    cand_attrs = {
        attr: [catalog[j][1][f"norm_{attr}"] for j in range(n)]
        for attr in attr_keys
    }
    cand_colors = [catalog[j][1]["norm_color"] for j in range(n)]

    for query_index in range(n):
        relevant_indexes = relevance_map[query_index]
        if not relevant_indexes:
            skipped_queries += 1
            continue

        query_id, query_row, _ = catalog[query_index]

        # Calculate scores for all candidates using fast lookups
        ranked_indexes = []
        q_attrs = {attr: query_row[f"norm_{attr}"] for attr in attr_keys}

        for candidate_index in range(n):
            if candidate_index == query_index:
                continue

            similarity = similarity_matrix[query_index, candidate_index]

            if visual_only:
                score = similarity * sim_weight
            else:
                metadata_score = 0
                for attr in attr_keys:
                    if q_attrs[attr] == cand_attrs[attr][candidate_index]:
                        metadata_score += meta_weights[attr]

                if max_metadata_score > 0:
                    metadata_score = metadata_score / max_metadata_score
                else:
                    metadata_score = 0

                score = (similarity * sim_weight) + (metadata_score * meta_influence)

            ranked_indexes.append((candidate_index, score))

        ranked_indexes.sort(
            key=lambda result: result[1],
            reverse=True,
        )

        evaluated_queries += 1
        total_relevant_items += len(relevant_indexes)
        relevant_items_seen = 0
        average_precision = 0.0

        for rank, (candidate_index, _) in enumerate(
            ranked_indexes,
            start=1,
        ):
            if candidate_index in relevant_indexes:
                relevant_items_seen += 1
                average_precision += relevant_items_seen / rank

        average_precision_total += average_precision / len(relevant_indexes)

        for k in recall_at:
            retrieved_indexes = {index for index, _ in ranked_indexes[:k]}
            recall_totals[k] += len(retrieved_indexes & relevant_indexes) / len(
                relevant_indexes
            )

        # Analyze errors in the top 5 retrieved candidates
        for candidate_index, _ in ranked_indexes[:5]:
            if candidate_index not in relevant_indexes:
                total_failures_analyzed += 1
                cand_row = catalog[candidate_index][1]
                
                cat_match = query_row["norm_category"] == cand_row["norm_category"]
                color_match = query_row["norm_color"] == cand_colors[candidate_index]
                
                if not cat_match:
                    wrong_category_count += 1
                    if color_match:
                        color_bias_count += 1
                    else:
                        # Check background bias: mismatch category, mismatch color, and zero matched features
                        matching_features = sum(
                            q_attrs[attr] == cand_attrs[attr][candidate_index]
                            for attr in attr_keys
                        )
                        if matching_features == 0:
                            background_bias_count += 1
                else:
                    # Category matches, but not relevant -> detail mismatch
                    missing_detail_count += 1

    if not evaluated_queries:
        raise ValueError("No queries have relevant items under the configured rule.")

    return {
        "catalog_size": len(catalog),
        "evaluated_queries": evaluated_queries,
        "skipped_queries": skipped_queries,
        "average_relevant_items": total_relevant_items / evaluated_queries,
        "mean_average_precision": average_precision_total / evaluated_queries,
        "recall": {k: recall_totals[k] / evaluated_queries for k in recall_at},
        "failures": {
            "wrong_category": (wrong_category_count / total_failures_analyzed * 100) if total_failures_analyzed > 0 else 0.0,
            "color_bias": (color_bias_count / total_failures_analyzed * 100) if total_failures_analyzed > 0 else 0.0,
            "missing_detail": (missing_detail_count / total_failures_analyzed * 100) if total_failures_analyzed > 0 else 0.0,
            "background_bias": (background_bias_count / total_failures_analyzed * 100) if total_failures_analyzed > 0 else 0.0,
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Systematic evaluation runner comparing visual search and multiple hybrid weights setups."
    )
    parser.add_argument(
        "--metadata",
        default="metadata.csv",
        help="Path to the catalog metadata CSV.",
    )
    parser.add_argument(
        "--embeddings",
        default="embeddings.pkl",
        help="Path to the generated image embeddings pickle.",
    )
    args = parser.parse_args()

    # Verify files exist
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file not found at {args.metadata}")
        return
    if not os.path.exists(args.embeddings):
        print(f"Error: Embeddings file not found at {args.embeddings}")
        return

    print("Loading catalog data...")
    raw_catalog = load_catalog(args.metadata, args.embeddings)
    
    # Run the optimizations
    catalog = pre_normalize_catalog(raw_catalog)
    relevance_map = precompute_relevance(catalog)
    
    recall_at = [1, 3, 5, 10]

    # Define the 5 configurations
    experiments = {
        "1. Baseline (Visual Only)": {
            "visual_only": True,
            "sim_weight": 1.0,
            "meta_influence": 0.0,
            "meta_weights": DEFAULT_WEIGHTS,
            "desc": "CLIP ViT-B/32 baseline visual-only similarity search."
        },
        "2. Default Hybrid": {
            "visual_only": False,
            "sim_weight": 0.80,
            "meta_influence": 0.20,
            "meta_weights": DEFAULT_WEIGHTS,
            "desc": "CLIP visual search with default metadata influence (20%)."
        },
        "3. High Metadata Influence": {
            "visual_only": False,
            "sim_weight": 0.60,
            "meta_influence": 0.40,
            "meta_weights": DEFAULT_WEIGHTS,
            "desc": "Stronger metadata influence (40%) and lower visual priority."
        },
        "4. Shape & Print Oriented": {
            "visual_only": False,
            "sim_weight": 0.70,
            "meta_influence": 0.30,
            "meta_weights": {
                "silhouette": 0.30,
                "pattern": 0.20,
                "style": 0.05,
                "neckline": 0.05,
                "sleeve": 0.05
            },
            "desc": "Prioritizes Silhouette/Shape (30%) and Pattern/Prints (20%)."
        },
        "5. Detail Oriented": {
            "visual_only": False,
            "sim_weight": 0.70,
            "meta_influence": 0.30,
            "meta_weights": {
                "neckline": 0.25,
                "sleeve": 0.25,
                "style": 0.05,
                "silhouette": 0.05,
                "pattern": 0.05
            },
            "desc": "Prioritizes garment component details (Neckline 25%, Sleeve 25%)."
        }
    }

    results = {}

    for name, config in experiments.items():
        print(f"\nRunning Experiment: {name}...")
        metrics = evaluate_opt(
            catalog,
            relevance_map,
            recall_at,
            visual_only=config["visual_only"],
            sim_weight=config["sim_weight"],
            meta_influence=config["meta_influence"],
            meta_weights=config["meta_weights"]
        )
        results[name] = metrics
        print(f"  mAP: {metrics['mean_average_precision']:.4f} | R@1: {metrics['recall'][1]:.4f} | R@10: {metrics['recall'][10]:.4f}")

    # Build Markdown Comparison Table
    report_lines = [
        "# Systematic Retrieval Performance Comparison Report\n",
        f"**Catalog Images:** {len(catalog)}",
        f"**Evaluated Queries:** {results['1. Baseline (Visual Only)']['evaluated_queries']}\n",
        "## Performance Metrics Comparison\n",
        "| Configuration | mAP | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Description |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    for name, metrics in results.items():
        desc = experiments[name]["desc"]
        r = metrics["recall"]
        line = f"| **{name}** | {metrics['mean_average_precision']:.4f} | {r[1]:.4f} | {r[3]:.4f} | {r[5]:.4f} | {r[10]:.4f} | {desc} |"
        report_lines.append(line)

    # Build Error Analysis markdown table
    report_lines.append("\n## Error Analysis & Failure Modes (Distribution within Top-5 Failures)\n")
    report_lines.append("| Configuration | Wrong Category | Color Bias | Background Bias | Missing Detail Similarity |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for name, metrics in results.items():
        f = metrics["failures"]
        line = f"| **{name}** | {f['wrong_category']:.1f}% | {f['color_bias']:.1f}% | {f['background_bias']:.1f}% | {f['missing_detail']:.1f}% |"
        report_lines.append(line)

    report_lines.append("\n### Failure Mode Definitions:")
    report_lines.append("- **Wrong Category**: The model retrieves a completely different garment type (e.g. retrieving pants when querying for a skirt) due to high-level visual/shape similarities.")
    report_lines.append("- **Color Bias**: The model retrieves an incorrect garment type that shares the same dominant color as the query (subset of Wrong Category).")
    report_lines.append("- **Background Bias**: Mismatches on category/color with zero matching tags, indicating similarity was heavily influenced by background, lighting, or model poses.")
    report_lines.append("- **Missing Detail Similarity**: The model correctly identifies the category but fails to match key design features (e.g. neckline, sleeves, pattern, style).")

    report_content = "\n".join(report_lines)

    # Output to Console
    print("\n" + "="*80)
    print(" EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(report_content)
    print("="*80)

    # Ensure Evaluation Results directory exists
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Evaluation Results")
    os.makedirs(output_dir, exist_ok=True)

    # Save Markdown Report to Output Directory
    report_filename = os.path.join(output_dir, "evaluation_report.md")
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\nSaved markdown report to: {os.path.abspath(report_filename)}")

    # Plot Recall Curve Comparison Graph
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', 'v']

    for i, (name, metrics) in enumerate(results.items()):
        recall_scores = [metrics["recall"][k] for k in recall_at]
        plt.plot(
            recall_at,
            recall_scores,
            marker=markers[i % len(markers)],
            linewidth=2.5,
            label=name
        )

    plt.title("Recall@K Curve Comparison Across Configurations", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("K (Number of Top Results Returned)", fontsize=12)
    plt.ylabel("Recall Score", fontsize=12)
    plt.xticks(recall_at)
    plt.ylim(0, 0.1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=10, frameon=True)
    plt.tight_layout()

    chart_filename = os.path.join(output_dir, "recall_comparison.png")
    plt.savefig(chart_filename, dpi=300)
    plt.close()
    print(f"Saved recall curve chart to: {os.path.abspath(chart_filename)}")

    # Copy chart to active conversation artifacts folder if running inside agent workspace
    artifacts_dir = r"C:\Users\vansh\.gemini\antigravity\brain\4c3f4e3b-9077-467e-a6f2-c0faa1bc62ce"
    if os.path.exists(artifacts_dir):
        try:
            shutil.copy(chart_filename, os.path.join(artifacts_dir, "recall_comparison.png"))
            print(f"Copied visual chart to conversation artifacts folder.")
        except Exception as e:
            print(f"Notice: Could not copy chart to artifacts directory: {e}")


if __name__ == "__main__":
    main()
