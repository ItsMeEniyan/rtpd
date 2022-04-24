from modules.hands import mainFunction as mainHands
from modules.body import mainFunction as mainBody

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

    age=request.headers['age']
    body_unicode = request.body
    f = open("test.mp4", "wb")
    f.write(body_unicode)
    f.close()
    
    print("body")
    result = mainBody(int(age))

    output= {
        'result': result,
        }

    return render(request, 'result.html', output)

def resulthand(request):

    age=request.headers['age']
    body_unicode = request.body
    f = open("test.mp4", "wb")
    f.write(body_unicode)
    f.close()
    
    print("hands")
    #call the function in hands.py
    result = mainHands(int(age))

    output= {
        'result': result,
        }

    return render(request, 'result.html', output)




