import pytest
import pandas as pd

from topn.preprocess.preprocessor import DataPreprocessor


# Unit tests for DataPreprocessor class

# Creating mock data for testing
@pytest.fixture
def mock_raw_data():
    data = pd.DataFrame({
        'userId': [1, 1, 2, 2, 3, 4, 5],
        'movieId': [101, 102, 101, 103, 102, None, 104],
        'rating': [5, 4, 3, 2, 1, 3, None]
    })

    return data

# Loading mock data for testing
@pytest.fixture
def sample_raw_data():
    data = pd.read_csv("tests/resources/ratings.csv")
    return data


def test_clean_data(mock_raw_data):
    preprocessor = DataPreprocessor(mock_raw_data)
    preprocessor.clean_data()
    assert mock_raw_data.shape == (5, 3)

def test_transform_data(mock_raw_data):
    preprocessor = DataPreprocessor(mock_raw_data)
    preprocessor.transform_data()
    assert mock_raw_data['userId'].dtype == 'int8'
    assert mock_raw_data['movieId'].dtype == 'int8'


def test_clean_data_2(sample_raw_data):
    preprocessor = DataPreprocessor(sample_raw_data)
    preprocessor.clean_data()
    assert sample_raw_data.shape == (20, 4)

def test_transform_data_2(sample_raw_data):
    preprocessor = DataPreprocessor(sample_raw_data)
    preprocessor.transform_data()
    assert sample_raw_data['userId'].dtype == 'int8'
    assert sample_raw_data['movieId'].dtype == 'int8'
