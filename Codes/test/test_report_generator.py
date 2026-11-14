import pandas as pd
from src.generate_final_report import md_table


def test_markdown_table_handles_empty_results():
    assert md_table(pd.DataFrame()) == "_No results available._"


def test_markdown_table_formats_numeric_values():
    result = md_table(pd.DataFrame([{"method": "pca", "mae_deg": 1.23456}]))
    assert "| pca | 1.235 |" in result
