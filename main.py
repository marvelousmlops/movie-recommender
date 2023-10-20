from topn.model.recommender import RecommenderSystem
from topn.utils.helper import load_data
from topn.preprocess.preprocessor import DataPreprocessor
from topn.evaluation.evaluator import RecommenderEvaluator


# Main script
if __name__ == "__main__":

    file_path = "data/ratings.csv"
    data = load_data(file_path)

    preprocessor = DataPreprocessor(data)
    preprocessor.clean_data()
    preprocessor.transform_data()
    processed_data = preprocessor.get_data()

    rec_sys = RecommenderSystem(processed_data)

    movie_id = 104

    top_recommendations = rec_sys.get_top_n_recommendations(movie_id)
    print(f"Top recommendations for movie ID {movie_id}: {top_recommendations}")

    
    gold_data = {
        101: [548, 60, 102, 531, 863],
        102: [247, 103, 270, 188, 523],
        103: [1514, 1967, 1843, 2770, 1233],
        104: [3810, 456, 2874, 133, 6637],
    }

    # Evaluate the recommender system
    evaluator = RecommenderEvaluator(rec_sys, gold_data)
    evaluation_results = evaluator.evaluate()
    print(f"Evaluation results: {evaluation_results}")
