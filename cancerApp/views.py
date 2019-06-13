from django.http import HttpResponse
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views import generic
from .forms import CustomUserCreationForm

# Create your views here.


# functions used for rendering pages each with its name (viewing them in html pages)


##########################################################
def index(request):
    return render(request, 'index.html')


def about(request):
    return render(request, 'about.html')


def visualize(request):
    return render(request, 'visualization.html')


def stats(request):
    return render(request, 'stats.html')


def login(request):
    return render(request, 'registration/login.html')


def base(request):
    return render(request, 'base.html')


def contact(request):
    return render(request, 'contact.html')


def diagnosis(request):
    return render(request, 'diagnosis.html')


class SignUp(generic.CreateView):
    form_class = CustomUserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'signup.html'

