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
        
def resultbody(request):

    body_unicode = request.body
    f = open("test.mp4", "wb")
    f.write(body_unicode)
    f.close()
    
    print("body")

    return render(request, 'result.html', {'result': 'yes'})

def resulthand(request):

    body_unicode = request.body
    f = open("test.mp4", "wb")
    f.write(body_unicode)
    f.close()
    
    print("hands")

    return render(request, 'result.html', {'result': 'yes'})