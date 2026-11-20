# movie-recommender

Repository for a simple recommender system, using a public ratings dataset.

## Popular movies and ratings by city

The project can report movies ranked by popularity within each city. Popularity
is measured by the number of ratings for a movie in that city; the report also
includes the average rating and rating count. Ties are ordered by average rating,
then title.

The ratings CSV must contain `userId`, `movieId`, and `rating`. Add two lookup
files alongside it:

- `data/movies.csv` with `movieId,title` columns.
- `data/users.csv` with `userId,city` columns.

IDs must match those in `data/ratings.csv`. These files should contain your
actual movie-title and user-city mappings. The example rows below are illustrative
only; the repository's ratings data does not provide city information, so do not
use the sample cities as real user locations.

```csv
movieId,title
1,Toy Story (1995)
2,Jumanji (1995)
3,Grumpier Old Men (1995)
```

```csv
userId,city
1,Seattle
2,Chicago
3,Seattle
```

When both lookup files are present, running `python main.py` prints the popular
movies for each city before running the existing recommendation and evaluation
steps. The ranking logic is also available as
`topn.model.city_popularity.get_popular_movies_by_city`; use `top_n` to control
how many movies are returned per city and `min_ratings` to require a minimum
number of ratings before a movie is included.
