import pytest
from src import preprocess

def test_load_and_clean_data():
    df = preprocess.load_and_clean_data()
    assert not df.empty, "Dataframe is empty"
    assert "tv_id" in df.columns, "Missing expected column 'tv_id'"
    assert "duration" in df.columns, "Missing expected column 'duration'"
