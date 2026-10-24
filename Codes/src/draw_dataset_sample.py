from pathlib import Path

import cv2
import numpy as np

def main():
    root = Path("data/DRASHTI-HaOBB")
    img_dir = root / "images" / "val"
    label_dir = root / "labels" / "val_original"
    out_path = Path("results/figures/dataset_sample.png")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )

    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        count = 0

        with open(label_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                vals = line.split()

                # 8 coordinates + class + difficulty + heading
                if len(vals) < 11:
                    continue

                try:
                    coords = [float(v) for v in vals[:8]]
                except ValueError:
                    continue

                pts = np.array(coords, dtype=np.float64).reshape(4, 2)
                pts = np.round(pts).astype(np.int32)

                # Draw the oriented bounding box in green.
                cv2.polylines(
                    img,
                    [pts.reshape((-1, 1, 2))],
                    True,
                    (0, 255, 0),
                    2,
                )

                # Add the vehicle class label.
                cls = vals[8]
                x = max(0, int(pts[:, 0].min()))
                y = max(15, int(pts[:, 1].min()) - 4)

                cv2.putText(
                    img, cls, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1,
                    cv2.LINE_AA,
                )

                count += 1

        if count == 0:
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if cv2.imwrite(str(out_path), img):
            print(f"Image: {img_path.name}")
            print(f"Boxes drawn: {count}")
            print(f"Saved: {out_path}")
            return

    raise SystemExit("No matching image with valid annotations was found.")


if __name__ == "__main__":
    main()
