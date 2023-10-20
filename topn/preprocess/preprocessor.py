# Author: Marvelous MLOps

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