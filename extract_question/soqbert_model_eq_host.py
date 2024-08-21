from fastapi import FastAPI
from pydantic import BaseModel
import re
from transformers import pipeline
import time
import uvicorn
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import sys

# Load the model and tokenizer
classifier = pipeline("text-classification", model="ilert/SoQbert")

app = FastAPI()

class TextInput(BaseModel):
    text: str

def is_question(result, threshold=0.5):
    return result['label'] == 'LABEL_1' and result['score'] > threshold

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def extract_questions(text):
    sentences = split_sentences(text)
    questions = []
    scores = []
    
    for sentence in sentences:
        result = classifier(sentence)[0]
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