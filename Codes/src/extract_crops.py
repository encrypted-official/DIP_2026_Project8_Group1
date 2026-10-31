"""Extract small vehicle crops and keep their ground-truth metadata beside them."""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.angle_utils import heading_to_axis180
from src.parse_annotations import read_vehicle_rows

def find_image(image_dir, stem):
    """Find an image by file stem, without assuming its extension."""
    for suffix in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        path = image_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None

def crop_box(image, points, pad_frac=0.10):
    """Make an axis-aligned crop around the OBB; do not rotate it upright."""
    pts = np.asarray(points, dtype=np.float32)
    h, w = image.shape[:2]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)

    pad_x = max(2, int((x1 - x0) * pad_frac))
    pad_y = max(2, int((y1 - y0) * pad_frac))

    left = max(0, int(np.floor(x0)) - pad_x)
    top = max(0, int(np.floor(y0)) - pad_y)
    right = min(w, int(np.ceil(x1)) + pad_x)
    bottom = min(h, int(np.ceil(y1)) + pad_y)

    if right <= left or bottom <= top:
        return None, None

    return image[top:bottom, left:right].copy(), (left, top, right, bottom)

def main():
    parser = argparse.ArgumentParser(description="Extract vehicle crops from DRASHTI-HaOBB.")
    parser.add_argument("--root", default="data/DRASHTI-HaOBB", help="Dataset root folder")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit-images", type=int, default=10, help="0 means all images")
    parser.add_argument("--out", default="crops/week2")
    parser.add_argument("--include-difficult", action="store_true",
                        help="Include boundary/difficult annotations (difficulty=1)")
    args = parser.parse_args()

    root = Path(args.root)
    image_dir = root / "images" / args.split
    label_dir_name = "train_original" if args.split == "train" else f"{args.split}_original"
    label_dir = root / "labels" / label_dir_name

    if not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(
            f"Could not find image/label folders.\nExpected:\n  {image_dir}\n  {label_dir}"
        )

    out_dir = Path(args.out)
    crop_dir = out_dir / "images"
    crop_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted(label_dir.glob("*.txt"))
    if args.limit_images > 0:
        label_files = label_files[:args.limit_images]

    meta_rows = []
    crop_no = 0
    missing_images = 0

    for label_path in label_files:
        image_path = find_image(image_dir, label_path.stem)
        if image_path is None:
            missing_images += 1
            print(f"No matching image for {label_path.name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read {image_path}")
            continue

        for obj in read_vehicle_rows(label_path):
            if obj["difficulty"] == 1 and not args.include_difficult:
                continue

            crop, bounds = crop_box(image, obj["points"])
            if crop is None or crop.size == 0:
                continue

            crop_no += 1
            crop_name = f"{args.split}_{crop_no:06d}.jpg"
            crop_path = crop_dir / crop_name
            if not cv2.imwrite(str(crop_path), crop):
                print(f"Could not save {crop_path}")
                continue

            left, top, right, bottom = bounds
            meta_rows.append({
                "crop_id": crop_path.stem,
                "crop_path": str(crop_path.as_posix()),
                "source_image": image_path.name,
                "source_label": label_path.name,
                "class_name": obj["class_name"],
                "difficulty": obj["difficulty"],
                "heading_deg": round(obj["heading_deg"], 4),
                "gt_axis_deg": round(heading_to_axis180(obj["heading_deg"]), 4),
                "x0": left, "y0": top, "x1": right, "y1": bottom,
                "crop_width": int(crop.shape[1]),
                "crop_height": int(crop.shape[0]),
                "crop_area": int(crop.shape[1] * crop.shape[0]),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metadata.csv"
    fieldnames = list(meta_rows[0].keys()) if meta_rows else [
        "crop_id", "crop_path", "source_image", "source_label", "class_name",
        "difficulty", "heading_deg", "gt_axis_deg", "x0", "y0", "x1", "y1",
        "crop_width", "crop_height", "crop_area"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"Images checked: {len(label_files)}")
    print(f"Vehicle crops saved: {len(meta_rows)}")
    print(f"Labels without a matching image: {missing_images}")
    print(f"Metadata: {csv_path}")

if __name__ == "__main__":
    main()