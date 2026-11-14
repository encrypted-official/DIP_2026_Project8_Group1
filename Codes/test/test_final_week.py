import numpy as np
from src.estimate_hybrid import distance_to_image_axis, estimate_axis_hybrid
from src.estimate_pca_roi import estimate_axis_pca_roi
from src.front_back import extract_end_features
from src.angle_utils import axis_error

def test_axis_distance():
    assert distance_to_image_axis(2)==2
    assert distance_to_image_axis(88)==2
    assert distance_to_image_axis(178)==2
    assert distance_to_image_axis(45)==45

def test_hybrid_routes_pca_near_axis():
    pred,route=estimate_axis_hybrid(None,15,pca_angle=8,hough_angle=42)
    assert pred==8 and route=="pca"

def test_hybrid_routes_hough_for_diagonal():
    pred,route=estimate_axis_hybrid(None,15,pca_angle=45,hough_angle=43)
    assert pred==43 and route=="hough_diagonal"

def test_hybrid_fallbacks():
    pred,route=estimate_axis_hybrid(None,15,pca_angle=None,hough_angle=27)
    assert pred==27 and route=="hough_fallback"
    pred,route=estimate_axis_hybrid(None,15,pca_angle=45,hough_angle=None)
    assert pred==45 and route=="pca_fallback"

def test_roi_returns_none_for_blank():
    assert estimate_axis_pca_roi(np.zeros((40,60,3),dtype=np.uint8)) is None

def test_front_back_features_are_symmetric_measurements_not_labels():
    im=np.zeros((40,80,3),dtype=np.uint8); im[:,20:60]=180
    f=extract_end_features(im,0)
    assert "mean_difference" in f and "left_edge_density" in f
    assert "feature_note" in f

def test_angle_error_undirected():
    assert axis_error(179,1)==2
