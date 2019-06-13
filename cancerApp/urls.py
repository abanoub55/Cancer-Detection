from django.urls import path
from . import appLogic
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
    path('prediction', appLogic.prediction),
    path('visualizeRib', appLogic.ribVisualize),
    path('visualizeLung', appLogic.lungStructure),
    path('cancerSpread', appLogic.cancer_spread),
    path('cancerStats', appLogic.cancerStats),
    path('genderStats', appLogic.genderStats),
    path('ageStats', appLogic.ageStats),
    path('clearHistory', appLogic.clearHistory),
]


