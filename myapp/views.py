from django.shortcuts import render
from myapp.rec_model.model_build import predict_books

def home(request):
    return render(request, "home.html")


def recommend_movies():
    similarity, results_df = predict_books(1000, 100000)
    return results_df.title[:10].tolist()


def recommendations(request):
    return render(request, "recommendations.html", {"books": recommend_movies()})