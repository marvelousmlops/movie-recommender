from topn.model.recommender import RecommenderSystem
from topn.utils.helper import load_data
from topn.preprocess.preprocessor import DataPreprocessor


# Main script
if __name__ == "__main__":

    file_path = "data/ratings.csv"
    data = load_data(file_path)
    
    preprocessor = DataPreprocessor(data)
    preprocessor.clean_data()
    preprocessor.transform_data()
    processed_data = preprocessor.get_data()

    rec_sys = RecommenderSystem(processed_data)

    movie_id = 101

    top_recommendations = rec_sys.get_top_n_recommendations(movie_id)
    
    print(f"Top recommendations for movie ID {movie_id}: {top_recommendations}")
   