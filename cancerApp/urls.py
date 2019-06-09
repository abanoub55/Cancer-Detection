from django.urls import path

from . import views

urlpatterns = [
    path('about', views.about, name='about'),
    path('contact', views.contact, name='contact'),
    path('base.html', views.base, name='base'),
    path('diagnosis', views.diagnosis, name='diagnosis'),
    path('visualization', views.visualize, name='visualize'),
    path('stats', views.stats, name='stats'),
    path('login', views.login, name='login'),
    path('signup', views.SignUp.as_view(), name='signup'),
    path('prediction', views.prediction),
    path('visualizeRib', views.ribVisualize),
    path('visualizeLung', views.lungStructure),
    path('cancerSpread', views.cancer_spread),
    path('cancerStats', views.cancerStats),
    path('genderStats', views.genderStats),
    path('ageStats', views.ageStats),
    path('clearHistory', views.clearHistory),
    path('confirm_upload',views.confirm_upload),
]



