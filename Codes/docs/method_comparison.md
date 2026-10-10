Week 2 method notes: PCA vs. Hough lines
Method A: PCA over edge pixels
Convert the crop to grayscale.
Apply light blur and CLAHE.
Run Canny edge detection.
Treat edge-pixel coordinates as a point cloud.
Find the covariance matrix and its dominant eigenvector.
Convert that vector into an undirected axis angle.
Why try it? It is a compact geometric baseline and needs no training.
Likely problems: road markings, shadows, windows, adjacent objects, and fragmented
edges can dominate the point cloud. A nearly round point cloud has no clear principal axis.
Method B: Hough-line orientation
Use the same preprocessing and Canny edge map.
Detect line segments with `HoughLinesP`.
Keep sufficiently long segments.
Combine segment angles with length weights.
Use doubled angles so opposite directions represent the same axis.
Why try it? Long vehicle edges may give useful orientation cues.
Likely problems: many vehicles have several strong edges; road lines may be longer
than vehicle edges; some crops may not contain a useful straight line.
Fair comparison
Use the exact same crop list for both methods.
Use the same ground-truth axis conversion and error function.
Report both the error and coverage (the percentage of crops that produced a prediction).
Do not hide invalid estimates. Record them as missing and report the valid-prediction count.
Do not claim PCA and Hough are independent learned models; both are classical geometry.
Week 2 conclusion template
Fill this after running the code:
Crops tested: ___
PCA valid predictions: ___
Hough valid predictions: ___
PCA MAE / median error: ___ / ___ degrees
Hough MAE / median error: ___ / ___ degrees
Classes or image conditions with high error: ___
Three example failure cases: ___
Next change to test: ___
