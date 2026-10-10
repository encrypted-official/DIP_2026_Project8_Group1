"""Read the simple per-image annotation files used by DRASHTI-HaOBB."""

from pathlib import Path
import re


def read_vehicle_rows(label_path):
    """Return annotation dictionaries from one label file.

    The first non-empty line is flight-height metadata, so it is skipped.
    The remaining rows contain four (x, y) corners, class, difficulty, heading.
    """
    label_path = Path(label_path)
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]

    if len(lines) <= 1:
        return []

    rows = []
    for line_no, line in enumerate(lines[1:], start=2):
        # The released files may use commas or spaces between values.
        parts = [part for part in re.split(r"[\s,]+", line) if part]
        if len(parts) < 11:
            print(f"Skipping {label_path.name}:{line_no}: expected 11 fields, got {len(parts)}")
            continue

        try:
            coords = [float(value) for value in parts[:8]]
            class_name = parts[8]
            difficulty = int(float(parts[9]))
            heading_deg = float(parts[10])
        except ValueError:
            print(f"Skipping {label_path.name}:{line_no}: could not parse row")
            continue

        pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
        rows.append({
            "points": pts,
            "class_name": class_name,
            "difficulty": difficulty,
            "heading_deg": heading_deg % 360.0,
            "label_file": label_path.name,
            "label_line": line_no,
        })

    return rows