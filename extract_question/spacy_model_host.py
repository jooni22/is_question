import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from spacy.matcher import Matcher
import uvicorn

# Załadowanie modelu spaCy
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

class TextInput(BaseModel):
    text: str

def extract_questions(text):
    doc = nlp(text)
    questions = []
    for sent in doc.sents:
        if sent.text.strip().endswith('?'):
            questions.append(sent.text.strip())
    return questions

@app.post("/classify")
async def classify_text(input: TextInput):
    questions = extract_questions(input.text)
    return {"questions": questions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8111)



import spacy


nlp = spacy.load("en_core_web_sm")


    