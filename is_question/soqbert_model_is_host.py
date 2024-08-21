from transformers import pipeline
from test_cases_iq import test_cases
import json

# Load the model and tokenizer
classifier = pipeline("text-classification", model="ilert/SoQbert")

# Evaluate model on test cases
correct = 0
total = len(test_cases)

for text, expected in test_cases:
    result = classifier(text)[0]
    
    if result['label'] == "question":
        output = {
            "questions": [text],
            "scores": [result['score']]
        }
    else:  # result['label'] == "statement"
        output = {
            "questions": [],
            "scores": []
        }
    
    print(json.dumps(output))
    print()