import onnxruntime
from transformers import AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import re
import uvicorn
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import sys
import requests

# Function to download ONNX model
def download_onnx_model(url, save_path):
    # Upewnij się, że katalog istnieje
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if not os.path.exists(save_path):
        print(f"Downloading ONNX model from {url}")
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded and saved to {save_path}")
    else:
        print(f"ONNX model already exists at {save_path}")

# URL of the ONNX model
onnx_model_url = "https://huggingface.co/ilert/SoQbert/resolve/main/onnx/model.onnx"
onnx_model_path = "onnx/model.onnx"

# Download the ONNX model
download_onnx_model(onnx_model_url, onnx_model_path)

# Load the tokenizer and ONNX model
tokenizer = AutoTokenizer.from_pretrained("ilert/SoQbert")
onnx_model = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

app = FastAPI()

class TextInput(BaseModel):
    text: str

def is_question(result, threshold=0.5):
    return result['label'] == 'LABEL_1' and result['score'] > threshold

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def classify_text_onnx(text):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
    onnx_inputs = {name: inputs[name] for name in ['input_ids', 'attention_mask']}
    logits = onnx_model.run(None, onnx_inputs)[0]
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    predicted_class = np.argmax(probabilities, axis=-1)[0]
    score = probabilities[0][predicted_class]
    label = "LABEL_1" if predicted_class == 1 else "LABEL_0"
    return {"label": label, "score": float(score)}

def extract_questions(text):
    sentences = split_sentences(text)
    questions = []
    scores = []
    
    for sentence in sentences:
        result = classify_text_onnx(sentence)
        if is_question(result) or '?' in sentence or sentence.lower().startswith(('what', 'why', 'how', 'when', 'where', 'who', 'which', 'whose', 'whom', 'can', 'could', 'would', 'should', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'may', 'might', 'will', 'shall')):
            questions.append(sentence)
            scores.append(result['score'])
    
    return questions, scores

@app.post("/classify")
async def classify_text(input: TextInput):
    questions, scores = extract_questions(input.text)
    return {"questions": questions, "scores": scores}

class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('soqbert_model_eq_host.py'):
            print("File change detected. Restarting server...")
            os.execv(sys.executable, ['python'] + sys.argv)

def run_server_with_auto_reload():
    observer = Observer()
    handler = FileChangeHandler()
    observer.schedule(handler, path='.', recursive=False)
    observer.start()

    try:
        uvicorn.run(app, host="0.0.0.0", port=8112)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    run_server_with_auto_reload()