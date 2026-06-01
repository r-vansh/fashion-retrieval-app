import csv
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "metadata.csv"
IMAGE_DIR = PROJECT_ROOT / "dataset" / "images"
OUTPUT_DIR = Path("C:/tmp/custom-metadata-audit")

BATCH_SIZE = 20
COLUMNS = 4
ROWS = 5
CELL_WIDTH = 430
CELL_HEIGHT = 430


def draw_wrapped(draw, text, x, y, font):
    for line in textwrap.wrap(text, width=60):
        draw.text((x, y), line, fill="black", font=font)
        y += 14

    return y


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with METADATA_PATH.open(newline="", encoding="utf-8-sig") as file:
        metadata = list(csv.DictReader(file))

    font = ImageFont.load_default()

    for batch_start in range(0, len(metadata), BATCH_SIZE):
        batch = metadata[batch_start:batch_start + BATCH_SIZE]
        sheet = Image.new(
            "RGB",
            (COLUMNS * CELL_WIDTH, ROWS * CELL_HEIGHT),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        for position, row in enumerate(batch):
            x = (position % COLUMNS) * CELL_WIDTH
            y = (position // COLUMNS) * CELL_HEIGHT

            with Image.open(IMAGE_DIR / row["file_name"]) as image:
                image = image.convert("RGB")
                image.thumbnail((300, 300))
                sheet.paste(
                    image,
                    (
                        x + (300 - image.width) // 2 + 5,
                        y + 5,
                    ),
                )

            label_y = y + 312
            labels = [
                f"{batch_start + position + 1:03d} {row['image_id']}",
                f"cat={row['category']} style={row['style']}",
                f"sil={row['silhouette']} sleeve={row['sleeve']}",
                f"neck={row['neckline']} pat={row['pattern']}",
            ]

            for label in labels:
                label_y = draw_wrapped(
                    draw,
                    label,
                    x + 8,
                    label_y,
                    font,
                )

        output_path = OUTPUT_DIR / f"batch_{batch_start // BATCH_SIZE + 1:02d}.jpg"
        sheet.save(output_path, quality=94)

    print(f"Generated {(len(metadata) + BATCH_SIZE - 1) // BATCH_SIZE} sheets")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
