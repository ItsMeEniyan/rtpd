from django.contrib import admin
from django.urls import path

from .views import home, resulthand, resultbody

urlpatterns = [
    path('', home, name='home'),
    path('analysehand/', resulthand, name='result'),
    path('analysebody/', resultbody, name='result')
]