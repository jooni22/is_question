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
    matcher = Matcher(nlp.vocab)
    
    # Wzorzec dla pytań rozpoczynających się od słów pytających
    question_words = ["what", "when", "where", "who", "whom", "which", "whose", "why", "how"]
    question_pattern = [{"LOWER": {"IN": question_words}}]
    matcher.add("QUESTION_WORD", [question_pattern])
    
    questions = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        
        # Sprawdzenie, czy zdanie kończy się znakiem zapytania
        if sent_text.endswith('?'):
            questions.append(sent_text)
        
        # Sprawdzenie, czy zdanie zaczyna się od słowa pytającego
        elif matcher(sent.as_doc()):
            questions.append(sent_text)
        
        # Sprawdzenie, czy zdanie zawiera słowo "question"
        elif "question" in sent_text.lower():
            questions.append(sent_text)
        
        # Sprawdzenie struktury zdania dla pytań bez znaku zapytania
        else:
            for token in sent:
                if token.dep_ == "aux" and token.i == 0:
                    questions.append(sent_text)
                    break
    
    return questions

@app.post("/classify")
async def classify_text(input: TextInput):
    questions = extract_questions(input.text)
    return {"questions": questions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8111)



import spacy


nlp = spacy.load("en_core_web_sm")


    