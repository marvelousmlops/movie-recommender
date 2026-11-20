import pandas as pd
import pytest

from topn.model.city_popularity import get_popular_movies_by_city


def test_popular_movies_ranked_by_city_rating_count_then_average():
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 1, 2, 3],
            "movieId": [10, 10, 20, 20, 10],
            "rating": [4.0, 5.0, 5.0, 4.0, 3.0],
        }
    )
    movies = pd.DataFrame({"movieId": [10, 20], "title": ["Alpha", "Beta"]})
    users = pd.DataFrame(
        {"userId": [1, 2, 3], "city": ["Seattle", "Seattle", "Chicago"]}
    )

    result = get_popular_movies_by_city(ratings, movies, users, top_n=2)

    seattle = result[result["city"] == "Seattle"].reset_index(drop=True)
    assert seattle["title"].tolist() == ["Alpha", "Beta"]
    assert seattle["rating_count"].tolist() == [2, 2]
    assert seattle["average_rating"].tolist() == [4.5, 4.5]
    assert result[result["city"] == "Chicago"]["title"].tolist() == ["Alpha"]


def test_minimum_rating_count_and_top_n_are_applied():
    ratings = pd.DataFrame({"userId": [1, 2], "movieId": [10, 10], "rating": [4, 5]})
    movies = pd.DataFrame({"movieId": [10], "title": ["Alpha"]})
    users = pd.DataFrame({"userId": [1, 2], "city": ["Seattle", "Seattle"]})

    result = get_popular_movies_by_city(ratings, movies, users, min_ratings=3)

    assert result.empty
    with pytest.raises(ValueError, match="top_n"):
        get_popular_movies_by_city(ratings, movies, users, top_n=0)


def test_missing_required_column_is_reported():
    ratings = pd.DataFrame({"userId": [1], "movieId": [10]})
    movies = pd.DataFrame({"movieId": [10], "title": ["Alpha"]})
    users = pd.DataFrame({"userId": [1], "city": ["Seattle"]})

    with pytest.raises(ValueError, match="rating"):
        get_popular_movies_by_city(ratings, movies, users)
