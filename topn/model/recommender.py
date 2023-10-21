# Author: Marvelous MLOps

from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity


class RecommenderSystem:
    def __init__(self, data):
        self.data = data
        self.user_movie_matrix = self._create_user_movie_matrix()
        self.movie_similarity_matrix = self._calculate_movie_similarity()

    def _create_user_movie_matrix(self):
        user_movie_matrix = self.data.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
        return user_movie_matrix

    def _calculate_movie_similarity(self):
        movie_similarity = cosine_similarity(self.user_movie_matrix.T)
        movie_similarity_matrix = DataFrame(movie_similarity, index=self.user_movie_matrix.columns, columns=self.user_movie_matrix.columns)
        return movie_similarity_matrix

    def get_top_n_recommendations(self, movie_id, n=5):
        similar_movies = self.movie_similarity_matrix[movie_id]
        recommended_movies = similar_movies.sort_values(ascending=False).head(n + 1).index.tolist()
        recommended_movies.remove(movie_id)
        return recommended_movies