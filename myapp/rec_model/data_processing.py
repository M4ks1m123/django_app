import math
import numpy as np
import pandas as pd
import torch

# def load_data():
#     ratings, books = pd.read_csv('C:/Users/Caster/Desktop/django_app/mysite/myapp/rec_model/data/Ratings.csv'), pd.read_csv('C:/Users/Caster/Desktop/django_app/mysite/myapp/rec_model/data/Books.csv')
    
#     books = books.drop(['Book-Author','Year-Of-Publication','Image-URL-S','Image-URL-M','Image-URL-L'], axis=1)
#     books = books.rename(columns={'ISBN':'bookId', 'Book-Title':'title', 'Publisher': 'genres'})
#     ratings = ratings.rename(columns={'User-ID': 'userId', 'ISBN': 'bookId', 'Book-Rating': 'rating'})
#     ratings['rating'] = pd.to_numeric(ratings['rating'], downcast='float')
#     ratings['rating'] = ratings['rating'].astype('float64')
    
#     return ratings, books
        
def create_dataset(ratings, top=None):
    if top is not None:
        ratings.groupby('userId')['rating'].count()

    unique_users = ratings.userId.unique()
    user_to_index = {old: new for new, old in enumerate(unique_users)}
    new_users = ratings.userId.map(user_to_index)

    unique_books = ratings.bookId.unique()
    # print(unique_books.size)
    book_to_index = {old: new for new, old in enumerate(unique_books)}
    new_books = ratings.bookId.map(book_to_index)
    # print(new_books.size)

    n_users = unique_users.shape[0]
    n_books = unique_books.shape[0]

    X = pd.DataFrame({'user_id': new_users, 'book_id': new_books})
    y = ratings['rating'].astype(np.float32)
    return (n_users, n_books), (X, y), (user_to_index, book_to_index)

class ReviewsIterator:

    def __init__(self, X, y, batch_size=32, shuffle=True):
        X, y = np.asarray(X), np.asarray(y)

        if shuffle:
            index = np.random.permutation(X.shape[0])
            X, y = X[index], y[index]

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = int(math.ceil(X.shape[0] // batch_size))
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        return self.X[k*bs:(k + 1)*bs], self.y[k*bs:(k + 1)*bs]

def batches(X, y, bs=32, shuffle=True):
    for xb, yb in ReviewsIterator(X, y, bs, shuffle):
        xb = torch.LongTensor(xb)
        yb = torch.FloatTensor(yb)
        yield xb, yb.view(-1, 1)
        
# ratings, books = load_data()

# print(ratings.size)
