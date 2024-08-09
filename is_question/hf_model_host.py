from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load tokenizer and model
model_name = "shahrukhx01/question-vs-statement-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Update model configuration
model.config.label2id = {"statement": 0, "question": 1}
model.config.id2label = {0: "statement", 1: "question"}

# Create text classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/classify")
async def classify_text(input: TextInput):
    try:
        result = classifier(input.text, truncation=True, max_length=512)
        is_question = result[0]['label'] == 'question'
        return {"text": input.text, "is_question": is_question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify_batch")
async def classify_batch(inputs: list[TextInput]):
    try:
        texts = [input.text for input in inputs]
        results = classifier(texts, truncation=True, max_length=512)
        classified_results = [
            {"text": text, "is_question": result['label'] == 'question'}
            for text, result in zip(texts, results)
        ]
        return classified_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
