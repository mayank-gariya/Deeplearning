from fastapi import FastAPI, UploadFile, File
from classifier import get_classification
from lookup import lookup

app = FastAPI()

@app.post('/predict_garbage')
async def garbage_classification(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    class_idx = get_classification(image_bytes)
    result = lookup.get(class_idx, "Unknown")
    
    return {
        'response': result,
        'status': 200
    }
