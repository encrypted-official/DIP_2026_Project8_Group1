import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.parse_annotations import read_vehicle_rows
from src.extract_crops import find_image
from src.angle_utils import heading_to_axis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/DRASHTI-HaOBB")
    parser.add_argument("--split", default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--out", default="results/week5_dataset_audit")
    args = parser.parse_args()

    root = Path(args.root)
    image_dir = root / "images" / args.split
    label_name = (
        "train_original" if args.split == "train"
        else f"{args.split}_original"
    )
    label_dir = root / "labels" / label_name

    if not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(f"Missing folders: {image_dir} or {label_dir}")

    label_files = sorted(label_dir.glob("*.txt"))
    annotations = []
    image_sizes = []
    missing = []
    unreadable = 0

    for label in label_files:
        image_path = find_image(image_dir, label.stem)

        if image_path is None:
            missing.append(label.name)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            unreadable += 1
            continue

        image_sizes.append((image.shape[1], image.shape[0]))

        for obj in read_vehicle_rows(label):
            annotations.append({
                "source_image": image_path.name,
                "class_name": obj["class_name"],
                "difficulty": obj["difficulty"],
                "heading_deg": obj["heading_deg"],
                "gt_axis_deg": heading_to_axis(obj["heading_deg"]),
            })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(annotations).to_csv(
        out / "annotation_audit.csv", index=False
    )

    if image_sizes:
        sizes, counts = np.unique(
            np.asarray(image_sizes), axis=0, return_counts=True
        )
        pd.DataFrame([
            {"width": int(w), "height": int(h), "images": int(n)}
            for (w, h), n in zip(sizes, counts)
        ]).to_csv(out / "image_dimensions.csv", index=False)

    report = [
        f"Split: {args.split}",
        f"Label files: {len(label_files)}",
        f"Readable paired images: {len(image_sizes)}",
        f"Missing image matches: {len(missing)}",
        f"Unreadable images: {unreadable}",
        f"Parsed annotations: {len(annotations)}",
    ]

    if args.metadata:
        meta = pd.read_csv(args.metadata)
        report.append(f"Crop metadata rows: {len(meta)}")

        if {"crop_width", "crop_height"}.issubset(meta.columns):
            short = meta[["crop_width", "crop_height"]].min(axis=1)
            report += [
                f"Short side <20 px: {(short < 20).sum()}",
                f"Short side 20-40 px: "
                f"{((short >= 20) & (short <= 40)).sum()}",
                f"Short side >40 px: {(short > 40).sum()}",
            ]

    (out / "audit_summary.txt").write_text(
        "\n".join(map(str, report)) + "\n", encoding="utf-8"
    )
    print("\n".join(map(str, report)))
    print(f"Saved audit to {out}")


if __name__ == "__main__":
    main()