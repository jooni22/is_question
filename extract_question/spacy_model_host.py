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

def is_rhetorical_question(text):
    doc = nlp(text)
    
    # Przykładowe reguły heurystyczne
    rhetorical_patterns = [
        [{"LOWER": "isn't"}, {"LOWER": "it"}, {"LOWER": "obvious"}],
        [{"LOWER": "who"}, {"LOWER": "would"}, {"LOWER": "have"}, {"LOWER": "thought"}],
        [{"LOWER": "do"}, {"LOWER": "i"}, {"LOWER": "really"}, {"LOWER": "need"}, {"LOWER": "to"}, {"LOWER": "explain"}],
    ]
    
    matcher = Matcher(nlp.vocab)
    for i, pattern in enumerate(rhetorical_patterns):
        matcher.add(f"RHETORICAL_{i}", [pattern])
    
    matches = matcher(doc)
    
    # Sprawdzenie, czy pytanie kończy się wykrzyknikiem
    if text.strip().endswith('?!'):
        return True
    
    # Sprawdzenie, czy pytanie zawiera charakterystyczne zwroty
    if matches:
        return True
    
    return False

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
        
        # Sprawdzenie, czy zdanie kończy się znakiem zapytania i nie jest retoryczne
        if sent_text.endswith('?') and not is_rhetorical_question(sent_text):
            questions.append(sent_text)
        
        # Sprawdzenie, czy zdanie zaczyna się od słowa pytającego i nie jest retoryczne
        elif matcher(sent.as_doc()) and not is_rhetorical_question(sent_text):
            questions.append(sent_text)
        
        # Sprawdzenie, czy zdanie zawiera słowo "question" i nie jest retoryczne
        elif "question" in sent_text.lower() and not is_rhetorical_question(sent_text):
            questions.append(sent_text)
        
        # Sprawdzenie struktury zdania dla pytań bez znaku zapytania i nie jest retoryczne
        else:
            for token in sent:
                if token.dep_ == "aux" and token.i == 0 and not is_rhetorical_question(sent_text):
                    questions.append(sent_text)
                    break
    
    return questions

@app.post("/classify")
async def classify_text(input: TextInput):
    questions = extract_questions(input.text)
    return {"questions": questions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8111)