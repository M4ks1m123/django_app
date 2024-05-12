from django.shortcuts import render, HttpResponse
from myapp.functions import get_todos

def home(request):
    return render(request, "home.html")

# def get_todos():
#     items = ['todo12', 'todo22', 'todo32']
#     return items

def todos(request):
    # items = ['todo1', 'todo2', 'todo3']
    return render(request, "todos.html", {"todos": get_todos()})