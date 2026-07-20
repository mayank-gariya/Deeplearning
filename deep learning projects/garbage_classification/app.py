from fastapi import FastAPI
from classifier import get_classification
from lookup import lookup

app = FastAPI()

@app.get('/predict_garbage')
def garbage_classification():
    response = get_classification()
    result = lookup.get(response)
    
    return {
        'reponse':result,
        'status':200
    }
    