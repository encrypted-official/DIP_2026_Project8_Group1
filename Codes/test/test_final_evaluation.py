import pandas as pd
from src.final_evaluation import axis_style

def test_axis_style_classification():
    assert axis_style(2, 15) == 'axis_aligned'
    assert axis_style(88, 15) == 'axis_aligned'
    assert axis_style(42, 15) == 'diagonal'
    assert axis_style(float('nan'), 15) == 'unknown'
