class RecommenderEvaluator:
    def __init__(self, recommender, gold_data):
        self.recommender = recommender
        self.gold_data = gold_data

    def evaluate(self):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for movie_id, similar_movies in self.gold_data.items():
            recommended_movies = self.recommender.get_top_n_recommendations(movie_id, n=len(similar_movies))
            for movie in recommended_movies:
                if movie in similar_movies:
                    true_positives += 1
                else:
                    false_positives += 1
            false_negatives += len([movie for movie in similar_movies if movie not in recommended_movies])
        
        accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        evaluation_results = {
            "F1_score": round(f1_score, 2),
            "Accuracy": round(accuracy,2)
        }

        return evaluation_results
