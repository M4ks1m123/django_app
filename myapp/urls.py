from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("recommendations/", views.recommendations, name="Recommendations"),
    path("recommendations_json/", views.recommendations_json, name="Recommendations")
]
