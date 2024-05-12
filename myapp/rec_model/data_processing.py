import io
import os
import math
import copy
import pickle
import zipfile
from textwrap import wrap
from pathlib import Path
from itertools import zip_longest
from collections import defaultdict
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler

def load_data():
    ratings, movies = pd.read_csv('C:/Users/Caster/Desktop/django_app/mysite/myapp/rec_model/data/Ratings.csv'), pd.read_csv('C:/Users/Caster/Desktop/django_app/mysite/myapp/rec_model/data/Books.csv')
    
    movies = movies.drop(['Book-Author','Year-Of-Publication','Image-URL-S','Image-URL-M','Image-URL-L'], axis=1)
    movies = movies.rename(columns={'ISBN':'movieId', 'Book-Title':'title', 'Publisher': 'genres'})
    ratings = ratings.rename(columns={'User-ID': 'userId', 'ISBN': 'movieId', 'Book-Rating': 'rating'})
    ratings['rating'] = pd.to_numeric(ratings['rating'], downcast='float')
    ratings['rating'] = ratings['rating'].astype('float64')
    
    return ratings, movies
        
def create_dataset(ratings, top=None):
    if top is not None:
        ratings.groupby('userId')['rating'].count()

    unique_users = ratings.userId.unique()
    user_to_index = {old: new for new, old in enumerate(unique_users)}
    new_users = ratings.userId.map(user_to_index)

    unique_movies = ratings.movieId.unique()
    # print(unique_movies.size)
    movie_to_index = {old: new for new, old in enumerate(unique_movies)}
    new_movies = ratings.movieId.map(movie_to_index)
    # print(new_movies.size)

    n_users = unique_users.shape[0]
    n_movies = unique_movies.shape[0]

    X = pd.DataFrame({'user_id': new_users, 'movie_id': new_movies})
    y = ratings['rating'].astype(np.float32)
    return (n_users, n_movies), (X, y), (user_to_index, movie_to_index)

ratings, movies = load_data()

# print(ratings.size)
