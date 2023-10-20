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

# Loading mock data for testing
@pytest.fixture
def sample_data():
    data = pd.read_csv("tests/resources/processed_ratings.csv")
    return data


# Integration test for RecommenderSystem with mock data
def test_recommender_system(mock_data):
    rec_sys = RecommenderSystem(mock_data)
    recommendations = rec_sys.get_top_n_recommendations(101, n=2)
    assert len(recommendations) == 2
    assert recommendations == [102, 103]


# Integration test for RecommenderSystem with sample data
def test_recommender_system_2(sample_data):
    rec_sys = RecommenderSystem(sample_data)
    recommendations = rec_sys.get_top_n_recommendations(3, n=5)
    assert len(recommendations) == 5
    assert recommendations == [0, 11, 18, 17, 16]