import os
from fastapi import FastAPI, UploadFile, File
from gradio_client import Client, handle_file

app = FastAPI()

HF_SPACE_URL = "realmanise/garbage_classifier"
client = Client(HF_SPACE_URL)

@app.post('/predict_garbage')
async def garbage_classification(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(await file.read())
        
    try:
        result = client.predict(
            img=handle_file(temp_file_path),
            api_name="/predict"
        )
        
        response_data = {
            'response': str(result),
            'status': 200
        }
    except Exception as e:
        response_data = {
            'error': f"Prediction failed: {str(e)}",
            'status': 500
        }
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return response_data

@app.get('/')
def home():
    return {
        "message": "garbage classifier is running perfectly!",
        'test_accuracy':97,
        'val_accuracy':92
    }