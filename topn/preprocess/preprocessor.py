# class DataPreprocessor:
#     def __init__(self, data_path):
#         self.data = pd.read_csv(data_path)
        
#     def drop_unnecessary_columns(self):
#         self.data = self.data.drop(['timestamp', 'genres'], axis=1)
        
#     def handle_missing_values(self):
#         self.data = self.data.dropna()

#     def create_user_item_matrix(self):
#         self.user_item_matrix = self.data.pivot_table(index='userId', columns='movieId', values='rating')

#     def get_user_item_matrix(self):
#         return self.user_item_matrix
    

from sklearn.metrics.pairwise import cosine_similarity


class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def clean_data(self):
        # Example: Drop duplicates and handle missing values
        self.data.drop_duplicates(inplace=True)
        self.data.dropna(subset=['userId', 'movieId', 'rating'], inplace=True)

    def transform_data(self):
        # Example: Convert userId and movieId to categorical values
        self.data['userId'] = self.data['userId'].astype('category').cat.codes
        self.data['movieId'] = self.data['movieId'].astype('category').cat.codes

    def get_data(self):
        return self.data