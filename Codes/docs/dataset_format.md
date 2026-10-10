# Dataset format and crop extraction

## What the dataset already gives us

DRASHTI-HaOBB provides four OBB corner points, vehicle class, difficulty flag,
and heading angle. We do not draw new boxes or manually enter angles.

The label text file begins with a flight-height line. Each later non-empty line is
expected to contain:

`x1 y1 x2 y2 x3 y3 x4 y4 class difficulty heading_angle`

The parser accepts commas or spaces between fields. If a downloaded version has a
different format, check a raw label file and update the parser rather than guessing.

## Crop policy

The extractor finds the smallest axis-aligned rectangle around the four OBB points,
adds 10% padding, clips the rectangle to the image boundary, and saves that crop.
It does not rotate the crop. That preserves the orientation in image coordinates.

The OBB is used for **localization and ground truth only**. The estimators receive the
saved crop pixels and do not receive the annotation corners or ground-truth angle.

## Angle policy

The dataset heading indicates the vehicle front. This prototype studies an undirected
axis, so it folds heading angles into 0–90 degrees:

`r = heading_deg % 180`
`axis_deg = min(r, 180 - r)`

That means 25° and 205° have the same axis target. It also means the output does not
tell us which end is the front.

## Difficulty flag

Difficulty 1 marks a vehicle cropped at the image boundary. By default, the crop script
skips those rows so the first experiment is simpler. Add `--include-difficult` to include
them in a later experiment and report that choice.
