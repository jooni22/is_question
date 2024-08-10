import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span
import re
import uvicorn
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import sys

# Załadowanie modelu spaCy
nlp = spacy.load("en_core_web_sm")
#en_core_web_lg
app = FastAPI()

class TextInput(BaseModel):
    text: str

def extract_questions(text):
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    
    # 1. Rozszerzenie listy słów pytających
    question_words = ["what", "when", "where", "who", "whom", "which", "whose", "why", "how", "whether", "if"]
    question_pattern = [{"LOWER": {"IN": question_words}}]
    matcher.add("QUESTION_WORD", [question_pattern])
    
    # 2. Dodanie wzorców dla złożonych i pośrednich pytań
    complex_patterns = [
        [{"LOWER": "do"}, {"LOWER": "you"}, {"LOWER": "mind"}, {"POS": "VERB", "OP": "?"}],
        [{"LOWER": "would"}, {"LOWER": "you"}, {"LOWER": "be"}, {"LOWER": "able"}, {"LOWER": "to"}],
        [{"LOWER": "i"}, {"LOWER": {"IN": ["wonder", "was", "am"]}}, {"LOWER": "wondering"}],
        [{"LOWER": "could"}, {"LOWER": "you"}, {"LOWER": "please"}],
        [{"LOWER": "i'd"}, {"LOWER": "like"}, {"LOWER": "to"}, {"LOWER": "know"}],
        [{"LOWER": "i"}, {"LOWER": "would"}, {"LOWER": "like"}, {"LOWER": "to"}, {"LOWER": "know"}],
        [{"LOWER": "can"}, {"LOWER": "you"}, {"LOWER": "tell"}, {"LOWER": "me"}],
    ]
    for i, pattern in enumerate(complex_patterns):
        matcher.add(f"COMPLEX_{i}", [pattern])
    
    # 3. Dodanie fraz charakterystycznych dla pytań pośrednich i idiomatycznych
    indirect_phrases = [
        "i'm curious about", "i'd be interested to know", "i'm wondering if",
        "any idea", "do you happen to know", "would you mind telling me",
        "care to explain", "enlighten me about", "fill me in on",
        "what are your thoughts on", "what's your take on", "how do you feel about",
        "i'd appreciate your input on", "could you shed some light on",
        "i'd love to hear your opinion on", "what do you make of",
        "i'm intrigued by", "i'm keen to understand", "i'm trying to figure out",
    ]
    patterns = [nlp.make_doc(text) for text in indirect_phrases]
    phrase_matcher.add("INDIRECT", patterns)
    
    questions = []
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        
        # Sprawdzenie, czy zdanie kończy się znakiem zapytania
        if sent_text.endswith('?'):
            questions.append(sent_text)
            continue
        
        # Sprawdzenie wzorców złożonych i pośrednich pytań
        matches = matcher(sent.as_doc())
        if matches:
            questions.append(sent_text)
            continue
        
        # Sprawdzenie fraz charakterystycznych dla pytań pośrednich
        phrase_matches = phrase_matcher(sent.as_doc())
        if phrase_matches:
            questions.append(sent_text)
            continue
        
        # Sprawdzenie struktury zdania dla pytań bez znaku zapytania
        first_word = sent[0].text.lower()
        modal_verbs = ["can", "could", "will", "would", "shall", "should", "may", "might", "must"]
        if first_word in modal_verbs or first_word in ["do", "does", "did", "is", "are", "was", "were", "have", "has", "had"]:
            questions.append(sent_text)
    
    return questions

@app.post("/classify")
async def classify_text(input: TextInput):
    questions = extract_questions(input.text)
    return {"questions": questions}

class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('spacy_model_host.py'):
            print("Wykryto zmianę w pliku. Restartowanie serwera...")
            os.execv(sys.executable, ['python'] + sys.argv)

def run_server_with_auto_reload():
    observer = Observer()
    handler = FileChangeHandler()
    observer.schedule(handler, path='.', recursive=False)
    observer.start()

    try:
        uvicorn.run(app, host="0.0.0.0", port=8111)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    run_server_with_auto_reload()