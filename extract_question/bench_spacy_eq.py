import time
import requests
import json

def load_test_cases(file_path):
    test_cases = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            test_cases.append((data['context'], data['question']))
    return test_cases

def save_errors_to_file(errors, file_path):
    with open(file_path, 'w') as file:
        for error in errors:
            json.dump(error, file)
            file.write('\n')

def run_benchmark():
    test_cases = load_test_cases('extract_question/optimized_dataset.jsonl')
    correct_predictions = 0
    total_time = 0
    results = []
    errors = []

    for context, expected_question in test_cases:
        start_time = time.time()
        
        # Wysłanie zapytania do API
        response = requests.post("http://localhost:8111/classify", json={"text": context})
        extracted_questions = response.json()["questions"]
        scores = response.json().get("scores", [])  # Używamy get() w przypadku, gdy "scores" nie jest zwracane
        
        end_time = time.time()

        # Normalizacja pytań przed porównaniem
        expected_question_normalized = expected_question.lower().rstrip('?.')
        extracted_questions_normalized = [q.lower().rstrip('?.') for q in extracted_questions]

        correct = expected_question_normalized in extracted_questions_normalized
        if correct:
            correct_predictions += 1
        else:
            errors.append({
                "context": context,
                "expected_question": expected_question,
                "extracted_questions": extracted_questions,
                "scores": scores
            })
        
        request_time = end_time - start_time
        total_time += request_time
        results.append((context, expected_question, extracted_questions, correct, request_time, scores))

    accuracy = correct_predictions / len(test_cases) * 100
    avg_response_time = total_time / len(test_cases) * 1000  # Convert to milliseconds

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average response time: {avg_response_time:.2f} ms")

    print("\nDetailed Results:")
    for context, expected, extracted, correct, response_time, scores in results:
        print(f"Context: {context}")
        print(f"Expected question: {expected}")
        print(f"Extracted questions: {extracted}")
        if scores:
            print(f"Scores: {scores}")
        print(f"Correct: {'Yes' if correct else 'No'}")
        print(f"Response time: {response_time*1000:.2f} ms")
        print()

    # Zapisz błędy do pliku
    save_errors_to_file(errors, 'extract_question/error_dataset.jsonl')
    print(f"Saved {len(errors)} errors to extract_question/error_dataset.jsonl")

if __name__ == "__main__":
    run_benchmark()