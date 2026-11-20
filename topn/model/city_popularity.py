from pandas import DataFrame


def get_popular_movies_by_city(ratings, movies, users, top_n=10, min_ratings=1):
    """Return each city's most-rated movies with their average rating.

    Popularity is measured by the number of ratings. Ties are broken by
    average rating, highest first. All inputs are expected to contain their
    respective ID columns: ratings (userId, movieId, rating), movies
    (movieId, title), and users (userId, city).
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if min_ratings < 1:
        raise ValueError("min_ratings must be at least 1")

    required_columns = {
        "ratings": (ratings, {"userId", "movieId", "rating"}),
        "movies": (movies, {"movieId", "title"}),
        "users": (users, {"userId", "city"}),
    }
    for name, (frame, required) in required_columns.items():
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{name} is missing required columns: {', '.join(sorted(missing))}"
            )

    joined = ratings.merge(users[["userId", "city"]], on="userId", how="inner")
    joined = joined.merge(movies[["movieId", "title"]], on="movieId", how="inner")
    joined = joined.dropna(subset=["city", "title", "rating"])

    ranked = (
        joined.groupby(["city", "movieId", "title"], as_index=False)
        .agg(rating_count=("rating", "count"), average_rating=("rating", "mean"))
    )
    ranked = ranked[ranked["rating_count"] >= min_ratings]
    ranked = ranked.sort_values(
        ["city", "rating_count", "average_rating", "title"],
        ascending=[True, False, False, True],
    )
    return ranked.groupby("city", sort=False, group_keys=False).head(top_n).reset_index(drop=True)
