import json
from django.shortcuts import render

def home(request):
    return render(request, 'index.html')

def getPredictions(video):

    #code here

    prediction = 0.0
    
    if prediction >= 50:
        return 'text'
    else:
        return 'error'
        
def result(request):

    body_unicode = request.body
    f = open("test.mp4", "wb")
    f.write(body_unicode)
    f.close()
    
    print("hai")

    return render(request, 'result.html', {'result': 'yes'})