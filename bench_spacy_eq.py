import spacy
import time
from test_cases_eq import test_cases

# Załadowanie modelu spaCy
nlp = spacy.load("en_core_web_sm")

def extract_questions(text):
    doc = nlp(text)
    questions = []
    for sent in doc.sents:
        if sent.text.strip().endswith('?'):
            questions.append(sent.text.strip())
    return questions

def run_benchmark():
    correct_predictions = 0
    total_time = 0
    results = []

    for text, expected_questions in test_cases:
        start_time = time.time()
        extracted_questions = extract_questions(text)
        end_time = time.time()

        correct = set(extracted_questions) == set(expected_questions)
        if correct:
            correct_predictions += 1
        
        request_time = end_time - start_time
        total_time += request_time
        results.append((text, expected_questions, extracted_questions, correct, request_time))

    accuracy = correct_predictions / len(test_cases) * 100
    avg_response_time = total_time / len(test_cases) * 1000  # Convert to milliseconds

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average response time: {avg_response_time:.2f} ms")

    print("\nDetailed Results:")
    for text, expected, extracted, correct, response_time in results:
        print(f"Text: {text}")
        print(f"Expected questions: {expected}")
        print(f"Extracted questions: {extracted}")
        print(f"Correct: {'Yes' if correct else 'No'}")
        print(f"Response time: {response_time*1000:.2f} ms")
        print()

if __name__ == "__main__":
    run_benchmark()