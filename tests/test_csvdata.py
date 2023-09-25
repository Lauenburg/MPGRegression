import io  # Add this import at the beginning of your file
from unittest.mock import patch

import pandas as pd
import pytest

from MPGRegression.data import CSVDataLoader  # replace with actual module name

# Sample CSV for testing
CSV_DATA = """
A,B,C
1,4,7
2,5,8
3,6,9
"""

# Sample URL for testing
TEST_URL = "http://test.com/data.csv"


@pytest.fixture
def csv_data_loader():
    return CSVDataLoader(
        csv_url=TEST_URL,
        csv_columns=["A", "B", "C"],
        csv_na_values=[],
        csv_comment="#",
        csv_sep=",",
        cat2hot_categories=[],
        cat2hot_mapping=[],
        split_frac=0.8,
        split_random_state=1,
        label_column="C",
    )


# Use the pytest fixture to inject the csv_data_loader object into test functions
def test_load(csv_data_loader):
    with patch("pandas.read_csv", return_value=pd.read_csv(io.StringIO(CSV_DATA))):
        csv_data_loader.load(
            url=TEST_URL, columns=["A", "B", "C"], na_values=[], comment="#", sep=","
        )
    assert not csv_data_loader.dataset.empty
    assert csv_data_loader.dataset.shape == (3, 3)


def test_one_hot_encode_columns(csv_data_loader):
    with patch("pandas.read_csv", return_value=pd.read_csv(io.StringIO(CSV_DATA))):
        csv_data_loader.load(
            url=TEST_URL, columns=["A", "B", "C"], na_values=[], comment="#", sep=","
        )
        csv_data_loader.one_hot_encode_columns(["A"], [["1", "2", "3"]])
    assert "1" in csv_data_loader.dataset.columns
    assert "2" in csv_data_loader.dataset.columns
    assert "3" in csv_data_loader.dataset.columns


def test_split_data(csv_data_loader):
    with patch("pandas.read_csv", return_value=pd.read_csv(io.StringIO(CSV_DATA))):
        csv_data_loader.load(
            url=TEST_URL, columns=["A", "B", "C"], na_values=[], comment="#", sep=","
        )
        csv_data_loader.split_data(frac=0.8, random_state=1)
    assert not csv_data_loader.train_dataset.empty
    assert not csv_data_loader.test_dataset.empty


def test_split_features_label(csv_data_loader):
    with patch("pandas.read_csv", return_value=pd.read_csv(io.StringIO(CSV_DATA))):
        csv_data_loader.load(
            url=TEST_URL, columns=["A", "B", "C"], na_values=[], comment="#", sep=","
        )
        csv_data_loader.split_data(frac=0.8, random_state=1)
        csv_data_loader.split_features_label(label_column="C")
    assert "C" not in csv_data_loader.train_features.columns
    assert not csv_data_loader.train_labels.empty
    assert "C" not in csv_data_loader.test_features.columns
    assert not csv_data_loader.test_labels.empty
