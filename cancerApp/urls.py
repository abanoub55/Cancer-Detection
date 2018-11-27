from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.upload, name='upload'),
    path('about.html', views.about, name='about'),
    path('contact.html', views.contact, name='contact'),
    path('base.html', views.base, name='base'),
    path('diagnosis.html', views.diagnosis, name='diagnosis'),
    path('visualization.html', views.visualize, name='visualize'),
    path('stats.html', views.stats, name='stats'),
    path('login', views.login, name='login'),
    path('signup.html', views.SignUp.as_view(), name='signup'),
]


