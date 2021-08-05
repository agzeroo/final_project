from django.urls import path
from . import views

urlpatterns = [
    # /step1/ 주소
    path('', views.index),
    # /step1/sub1/
    path('sub1/', views.sub1),
]