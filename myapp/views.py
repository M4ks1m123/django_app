from django.shortcuts import render, HttpResponse
from myapp.functions import get_todos
from myapp.rec_model.model_build import predict_movies

def home(request):
    return render(request, "home.html")

# def get_todos():
#     items = ['todo12', 'todo22', 'todo32']
#     return items


def recommend_movies():
    similarity, results_df = predict_movies('movie_embed', 8, 10, 100000)
    return results_df.title[:10].tolist()

def todos(request):
    # items = ['todo1', 'todo2', 'todo3']
    return render(request, "todos.html", {"todos": recommend_movies()})