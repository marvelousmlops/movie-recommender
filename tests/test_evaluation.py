import pytest
import pandas as pd

from topn.evaluation.evaluator import RecommenderEvaluator
from topn.model.recommender import RecommenderSystem

# Unit tests for Evaluator class

# Creating mock data for testing
@pytest.fixture
def mock_gold_data():
    gold_data = {
        3: [0, 11, 18, 17, 16],
        11: [7, 1, 18, 17, 16],
        16: [11, 1, 12, 17, 15],
        17: [0, 3, 18, 16, 15]
    }

    return gold_data

# Loading mock data for testing
@pytest.fixture
def sample_raw_data():
    data = pd.read_csv("tests/resources/processed_ratings.csv")
    return data


def test_evaluator(sample_raw_data, mock_gold_data):
    rec_sys = RecommenderSystem(sample_raw_data)
    evaluator = RecommenderEvaluator(rec_sys, mock_gold_data)
    evaluation_results = evaluator.evaluate()

    assert evaluation_results['F1_score'] == 0.80
    assert evaluation_results['Accuracy'] == 0.67
