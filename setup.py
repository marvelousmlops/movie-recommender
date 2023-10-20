from setuptools import setup, find_packages

__version__ = "0.0.1"

setup(
    name="topn",
    version=__version__,
    description="TopN Recommender System",
    author="Marvelous MLOps",
    packages=find_packages(exclude=["tests"]),
    zip_safe=False,
)
