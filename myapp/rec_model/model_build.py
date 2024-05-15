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
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler

from myapp.rec_model.model_config import EmbeddingNet

from myapp.rec_model.data_processing import load_data, create_dataset
from sklearn.metrics.pairwise import cosine_similarity

ratings, movies = load_data()

(n, m), (X, y), _ = create_dataset(ratings)
# print(f'Embeddings: {n} users, {m} movies')
# print(f'Dataset shape: {X.shape}')
# print(f'Target shape: {y.shape}')

net = EmbeddingNet(
    n_users=n, n_movies=m,
    n_factors=50, hidden=[500],
    embedding_dropout=0.05, dropouts=[0.25])

# print(net)
# MODEL_PATH = Path("models")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)

# MODEL_NAME = 'net_model.pth'
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

net.load_state_dict(torch.load(f='C:/Users/Caster/Desktop/django_app/mysite/myapp/rec_model/model_state_dict/net_model.pth'))
print('Successfully loaded state dict!')

tmp_movieIds = ratings.movieId.unique()
movie_set = {}
user_set = {}
for count, movie in enumerate(tmp_movieIds):
    movie_set[movie] = count

tmp_userIds = ratings.userId.unique()
for count, user in enumerate(tmp_userIds):
    user_set[user] = count

inv_movie_set = {v:k for k, v in movie_set.items()}
inv_user_set = {v:k for k, v in user_set.items()}

def predict_movies(embedding, idx, movies_df, n_movies):

    all_embedings = net.m(torch.arange(0, n_movies)).tolist()
    # print(all_embedings)
    index = (net.u(torch.tensor([idx])).detach().numpy())
    # print(f'index: {index.shape}')

    similarity = cosine_similarity(index, all_embedings)[0]
    # print(similarity)

    top = np.where(similarity > 0.7)[0]
    # print(f'top: {top.shape}')
    # top = np.delete(top, np.where(top == idx))

    scores = similarity[top]
    # print(all_embedings[:5])
    # print(top.shape)
    embedding_id = [inv_movie_set[result] for result in top]
    # print(f'Embedding id: {embedding_id}')
    titles = movies.loc[movies['movieId'].isin(embedding_id)]
    # print(f'titels: {titles.shape}')

    result_df = titles[['movieId','title']]
    # print(titles)
    # print(f'scores: {scores.shape}')
    # print(f'result_df: {result_df.shape}')
    # result_df['score'] = scores
    # result_df.sort_values(by='score', ascending=False, inplace=True)

    return similarity, result_df

similarity, results_df = predict_movies('movie_embed', 8, 10, 100000)

# print(results_df.title[:10].tolist())