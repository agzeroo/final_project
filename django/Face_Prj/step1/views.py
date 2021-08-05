from django.shortcuts import render
from django.http import HttpResponse 
# Create your views here.


def index(request):
    return HttpResponse("<h1>하이요 홈페이지임</h1>")

def sub1(request):
    return render(request, 'step1/sub1.html')