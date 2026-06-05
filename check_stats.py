import pandas as pd

df = pd.read_csv("metadata.csv")
fields = ["category", "silhouette", "sleeve", "neckline", "color", "style", "pattern"]

print("=== Per-field 'none' counts ===")
for f in fields:
    none_count = sum(1 for v in df[f] if str(v).strip().lower() == "none")
    print(f"  {f}: {none_count}")

print()
print("=== Per-field truly empty/NaN counts ===")
for f in fields:
    empty_count = sum(1 for v in df[f] if pd.isna(v) or str(v).strip() == "")
    print(f"  {f}: {empty_count}")

print()
total = len(df)
no_empty = sum(
    1 for _, r in df.iterrows()
    if not any(pd.isna(r.get(f, "")) or str(r.get(f, "")).strip() == "" for f in fields)
)
print(f"Total rows: {total}")
print(f"Rows with ALL fields filled (including 'none'): {no_empty}")
print(f"Rows with at least 1 truly empty field: {total - no_empty}")
