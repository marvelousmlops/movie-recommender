# Author: Marvelous MLOps

from pandas import read_csv

def load_data(file_path):
    data = read_csv(file_path)
    return data
