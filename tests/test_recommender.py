# Unit tests for preprocessing functions
import pytest
import pandas as pd

from topn.model.recommender import RecommenderSystem


@pytest.fixture
def mock_data():
    data = pd.DataFrame({
        'userId': [1, 1, 2, 2, 3],
        'movieId': [101, 102, 101, 103, 102],
        'rating': [5, 4, 3, 2, 1]
    })
    return data

def test_load_data(mock_data):
    assert mock_data.shape == (5, 3)

# Integration test for RecommenderSystem
def test_recommender_system(mock_data):
    rec_sys = RecommenderSystem(mock_data)
    recommendations = rec_sys.get_top_n_recommendations(101, n=2)
    assert len(recommendations) == 2