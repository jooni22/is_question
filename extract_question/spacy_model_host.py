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

def is_question_structure(sent):
    # Sprawdź, czy zdanie ma strukturę pytania
    root = [token for token in sent if token.dep_ == 'ROOT'][0]
    if root.pos_ == 'VERB' and root.i == 0:  # Czasownik na początku zdania
        return True
    if any(token.dep_ == 'aux' and token.i == 0 for token in sent):  # Czasownik pomocniczy na początku
        return True
    return False

def score_question(sent, matcher, phrase_matcher):
    score = 0
    sent_text = sent.text.lower()

    # Przyznaj punkty za różne cechy
    if sent_text.endswith('?'):
        score += 5
    if sent[0].text.lower() in ["what", "when", "where", "who", "whom", "which", "whose", "why", "how", "whether", "if"]:
        score += 4
    if is_question_structure(sent):
        score += 3
    if matcher(sent.as_doc()):
        score += 2
    if phrase_matcher(sent.as_doc()):
        score += 2
    if sent[0].text.lower() in ["can", "could", "will", "would", "shall", "should", "may", "might", "must", "do", "does", "did", "is", "are", "was", "were", "have", "has", "had"]:
        score += 1

    # Dodajemy punkty za słowa kluczowe często występujące w pośrednich pytaniach
    keywords = ["tell", "explain", "describe", "elaborate", "share", "wonder", "curious", "interested"]
    if any(token.lower_ in keywords for token in sent):
        score += 2

    return score

def extract_questions(text, threshold=3):
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    
    # 1. Wzorce dla złożonych i pośrednich pytań
    complex_patterns = [
        [{"LOWER": "do"}, {"LOWER": "you"}, {"LOWER": "mind"}, {"POS": "VERB", "OP": "?"}],
        [{"LOWER": "would"}, {"LOWER": "you"}, {"LOWER": "be"}, {"LOWER": "able"}, {"LOWER": "to"}],
        [{"LOWER": "i"}, {"LOWER": {"IN": ["wonder", "was", "am"]}}, {"LOWER": "wondering"}],
        [{"LOWER": "could"}, {"LOWER": "you"}, {"LOWER": "please"}],
        [{"LOWER": "i'd"}, {"LOWER": "like"}, {"LOWER": "to"}, {"LOWER": "know"}],
        [{"LOWER": "i"}, {"LOWER": "would"}, {"LOWER": "like"}, {"LOWER": "to"}, {"LOWER": "know"}],
        [{"LOWER": "can"}, {"LOWER": "you"}, {"LOWER": "tell"}, {"LOWER": "me"}],
        [{"LOWER": {"IN": ["tell", "explain", "describe", "elaborate", "share"]}}, {"OP": "*"}],
        [{"LOWER": "let"}, {"LOWER": "me"}, {"LOWER": "know"}],
        [{"LOWER": {"IN": ["wonder", "curious", "interested"]}}, {"OP": "*"}],
    ]
    for i, pattern in enumerate(complex_patterns):
        matcher.add(f"COMPLEX_{i}", [pattern])
    
    # 2. Frazy charakterystyczne dla pytań pośrednich i idiomatycznych
    indirect_phrases = [
        "i'm curious about", "i'd be interested to know", "i'm wondering if",
        "any idea", "do you happen to know", "would you mind telling me",
        "care to explain", "enlighten me about", "fill me in on",
        "what are your thoughts on", "what's your take on", "how do you feel about",
        "i'd appreciate your input on", "could you shed some light on",
        "i'd love to hear your opinion on", "what do you make of",
        "i'm intrigued by", "i'm keen to understand", "i'm trying to figure out",
        "tell me", "explain to me", "describe for me", "elaborate on",
        "share your thoughts", "let me know", "i wonder", "i'm curious about",
        "i'd like to know", "i'm interested in", "what are your thoughts on",
        "what's your opinion on", "how do you feel about",
    ]
    patterns = [nlp.make_doc(text) for text in indirect_phrases]
    phrase_matcher.add("INDIRECT", patterns)
    
    questions = []
    
    for sent in doc.sents:
        score = score_question(sent, matcher, phrase_matcher)
        
        # Zwiększamy wagę dla zdań zaczynających się od czasowników w trybie rozkazującym
        if sent[0].pos_ == "VERB" and sent[0].dep_ == "ROOT":
            score += 2
        
        if score >= threshold:
            questions.append((sent.text.strip(), score))
    
    return sorted(questions, key=lambda x: x[1], reverse=True)

@app.post("/classify")
async def classify_text(input: TextInput):
    questions = extract_questions(input.text)
    return {"questions": [q[0] for q in questions], "scores": [q[1] for q in questions]}

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