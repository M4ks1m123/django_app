import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from myapp.rec_model.model_config import RecommendationModel
from myapp.rec_model.data_processing import create_dataset
from myapp.rec_model.load_data import load_data

ratings, books = load_data()

(n, m), (X, y), _ = create_dataset(ratings)

net = RecommendationModel(
    n_users=n, n_books=m,
    n_factors=50, hidden=[500],
    embedding_dropout=0.05, dropouts=[0.25])


net.load_state_dict(torch.load(f='C:/Users/Caster/Desktop/django_app/mysite/myapp/rec_model/model_state_dict/net_model.pth'))
print('Successfully loaded state dict!')

tmp_bookIds = ratings.bookId.unique()
book_set = {}
user_set = {}
for count, book in enumerate(tmp_bookIds):
    book_set[book] = count

tmp_userIds = ratings.userId.unique()
for count, user in enumerate(tmp_userIds):
    user_set[user] = count

inv_book_set = {v:k for k, v in book_set.items()}
inv_user_set = {v:k for k, v in user_set.items()}

def predict_books(idx, n_books):

    all_embedings = net.m(torch.arange(0, n_books)).tolist()
    index = (net.u(torch.tensor([idx])).detach().numpy())

    similarity = cosine_similarity(index, all_embedings)[0]

    top = np.where(similarity > 0.7)[0]

    scores = similarity[top]

    embedding_id = [inv_book_set[result] for result in top]

    titles = books.loc[books['bookId'].isin(embedding_id)]


    result_df = titles[['bookId','title']]


    return similarity, result_df

similarity, results_df = predict_books(idx=8, n_books=100000)