import argparse
import os
import pickle

import pandas as pd
import torch


METADATA_WEIGHTS = {
    "style": 0.04,
    "silhouette": 0.03,
    "neckline": 0.015,
    "sleeve": 0.015,
    "pattern": 0.015,
}
SIMILARITY_WEIGHT = 0.90


def normalize(value):
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def normalize_category(value):
    return normalize(value).replace("pants", "pant")


def image_id_from_path(image_path):
    return os.path.splitext(
        os.path.basename(image_path)
    )[0].strip()


def load_catalog(metadata_path, embeddings_path):
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.set_index("image_id", drop=False)

    with open(embeddings_path, "rb") as file:
        image_paths, image_embeddings = pickle.load(file)

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


def count_matching_attributes(query_row, candidate_row):
    return sum(
        normalize(query_row[attribute])
        == normalize(candidate_row[attribute])
        for attribute in METADATA_WEIGHTS
    )


def is_relevant(query_row, candidate_row):
    return (
        normalize_category(query_row["category"])
        == normalize_category(candidate_row["category"])
        and count_matching_attributes(query_row, candidate_row) >= 3
    )


def hybrid_score(similarity, query_row, candidate_row, visual_only):
    score = similarity * SIMILARITY_WEIGHT

    if visual_only:
        return score

    for attribute, weight in METADATA_WEIGHTS.items():
        if (
            normalize(query_row[attribute])
            == normalize(candidate_row[attribute])
        ):
            score += weight

    return score


def evaluate(catalog, recall_at, visual_only):
    embeddings = torch.stack(
        [embedding for _, _, embedding in catalog]
    )
    embeddings = embeddings / embeddings.norm(
        dim=1,
        keepdim=True
    )
    similarity_matrix = embeddings @ embeddings.T

    recall_totals = {
        k: 0.0
        for k in recall_at
    }
    evaluated_queries = 0
    skipped_queries = 0
    total_relevant_items = 0
    average_precision_total = 0.0

    for query_index, (_, query_row, _) in enumerate(catalog):
        relevant_indexes = {
            candidate_index
            for candidate_index, (_, candidate_row, _) in enumerate(catalog)
            if candidate_index != query_index
            and is_relevant(query_row, candidate_row)
        }

        if not relevant_indexes:
            skipped_queries += 1
            continue

        ranked_indexes = []

        for candidate_index, (_, candidate_row, _) in enumerate(catalog):
            if candidate_index == query_index:
                continue

            similarity = similarity_matrix[
                query_index,
                candidate_index
            ].item()
            score = hybrid_score(
                similarity,
                query_row,
                candidate_row,
                visual_only,
            )
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

        average_precision_total += (
            average_precision / len(relevant_indexes)
        )

        for k in recall_at:
            retrieved_indexes = {
                index
                for index, _ in ranked_indexes[:k]
            }
            recall_totals[k] += (
                len(retrieved_indexes & relevant_indexes)
                / len(relevant_indexes)
            )

    if not evaluated_queries:
        raise ValueError(
            "No queries have relevant items under the configured rule."
        )

    return {
        "catalog_size": len(catalog),
        "evaluated_queries": evaluated_queries,
        "skipped_queries": skipped_queries,
        "average_relevant_items": (
            total_relevant_items / evaluated_queries
        ),
        "mean_average_precision": (
            average_precision_total / evaluated_queries
        ),
        "recall": {
            k: recall_totals[k] / evaluated_queries
            for k in recall_at
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fashion retrieval quality with Recall@K and mAP."
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
    parser.add_argument(
        "--recall-at",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        metavar="K",
        help="Recall cutoffs to calculate. Default: 1 3 5 10.",
    )
    parser.add_argument(
        "--visual-only",
        action="store_true",
        help="Evaluate similarity ranking without metadata boosts.",
    )

    args = parser.parse_args()

    if any(k <= 0 for k in args.recall_at):
        parser.error("Recall cutoffs must be positive integers.")

    args.recall_at = sorted(set(args.recall_at))

    return args


def main():
    args = parse_args()
    catalog = load_catalog(
        args.metadata,
        args.embeddings,
    )
    results = evaluate(
        catalog,
        args.recall_at,
        args.visual_only,
    )

    mode = "visual similarity only" if args.visual_only else "hybrid score"

    print(f"Mode: {mode}")
    print(
        "Relevant: same category and at least 3 matching metadata "
        f"attributes ({', '.join(METADATA_WEIGHTS)})"
    )
    print(f"Catalog images: {results['catalog_size']}")
    print(f"Evaluated queries: {results['evaluated_queries']}")
    print(f"Skipped queries without relevant items: {results['skipped_queries']}")
    print(
        "Average relevant items per evaluated query: "
        f"{results['average_relevant_items']:.2f}"
    )
    print(f"mAP: {results['mean_average_precision']:.4f}")

    for k, recall in results["recall"].items():
        print(f"Recall@{k}: {recall:.4f}")


if __name__ == "__main__":
    main()
