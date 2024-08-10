import time
import requests
from test_cases_eq import test_cases

def run_benchmark():
    correct_predictions = 0
    total_time = 0
    results = []

    for text, expected in test_cases:
        start_time = time.time()
        
        # Wysłanie zapytania do API
        response = requests.post("http://localhost:8111/classify", json={"text": text})
        extracted_questions = response.json()["questions"]
        scores = response.json()["scores"]
        
        end_time = time.time()

        is_question = len(extracted_questions) > 0
        correct = is_question == (len(expected) > 0)
        if correct:
            correct_predictions += 1
        
        request_time = end_time - start_time
        total_time += request_time
        results.append((text, expected, is_question, correct, request_time, extracted_questions, scores))

    accuracy = correct_predictions / len(test_cases) * 100
    avg_response_time = total_time / len(test_cases) * 1000  # Convert to milliseconds

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average response time: {avg_response_time:.2f} ms")

    print("\nDetailed Results:")
    for text, expected, predicted, correct, response_time, questions, scores in results:
        print(f"Text: {text}")
        print(f"Expected: {expected}")
        print(f"Predicted questions: {questions}")
        print(f"Scores: {scores}")
        print(f"Correct: {'Yes' if correct else 'No'}")
        print(f"Response time: {response_time*1000:.2f} ms")
        print()

if __name__ == "__main__":
    run_benchmark()