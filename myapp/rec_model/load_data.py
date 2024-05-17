import pandas as pd

def load_data():
    ratings, books = pd.read_csv('C:/Users/Caster/Desktop/django_app/mysite/myapp/rec_model/data/Ratings.csv', dtype='unicode'), pd.read_csv('C:/Users/Caster/Desktop/django_app/mysite/myapp/rec_model/data/Books.csv', dtype='unicode')
    
    books = books.drop(['Book-Author','Year-Of-Publication','Image-URL-S','Image-URL-M','Image-URL-L'], axis=1)
    books = books.rename(columns={'ISBN':'bookId', 'Book-Title':'title', 'Publisher': 'genres'})
    ratings = ratings.rename(columns={'User-ID': 'userId', 'ISBN': 'bookId', 'Book-Rating': 'rating'})
    ratings['rating'] = pd.to_numeric(ratings['rating'], downcast='float')
    ratings['rating'] = ratings['rating'].astype('float64')
    
    return ratings, books

ratings, books = load_data()

print(ratings.size)