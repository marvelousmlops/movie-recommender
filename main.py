from topn.model.recommender import RecommenderSystem
from topn.utils.helper import load_data
from topn.preprocess.preprocessor import DataPreprocessor


# Main script
if __name__ == "__main__":
    file_path = "data/ratings.csv"

    # data = load_data(file_path)
    # preprocessor = DataPreprocessor(data)
    # preprocessor.clean_data()
    # preprocessor.transform_data()
    # processed_data = preprocessor.get_data()

    # rec_sys = RecommenderSystem(processed_data)

    # movie_id = 101

    # top_recommendations = rec_sys.get_top_n_recommendations(movie_id)
    
    # print(f"Top recommendations for movie ID {movie_id}: {top_recommendations}")


    import pandas as pd

    data = pd.DataFrame({
        'userId': [1, 1, 2, 2, 3, 4, 5],
        'movieId': [101, 102, 101, 103, 102, None, 104],
        'rating': [5, 4, 3, 2, 1, 3, None]
    })
    
    
    print(data['userId'].dtype)
    print(data['movieId'].dtype)

    
    preprocessor = DataPreprocessor(data)
    preprocessor.clean_data()

    preprocessor.transform_data()


    print(preprocessor.data['userId'].dtype)
    print(preprocessor.data['movieId'].dtype)


