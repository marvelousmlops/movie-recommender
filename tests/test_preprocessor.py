import pytest
import pandas as pd

from topn.preprocess.preprocessor import DataPreprocessor


# Unit tests for DataPreprocessor class
@pytest.fixture
def mock_raw_data():
    data = pd.DataFrame({
        'userId': [1, 1, 2, 2, 3, 4, 5],
        'movieId': [101, 102, 101, 103, 102, None, 104],
        'rating': [5, 4, 3, 2, 1, 3, None]
    })

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
