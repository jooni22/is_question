import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from is_question.test_cases_iq import test_cases

API_URL = "http://localhost:8000/classify"

def send_request(text):
    start_time = time.time()
    response = requests.post(API_URL, json={"text": text})
    end_time = time.time()
    return response.json(), end_time - start_time

def run_benchmark():
    correct_predictions = 0
    total_time = 0
    results = []
    incorrect_results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_text = {executor.submit(send_request, text): (text, expected) for text, expected in test_cases}
        for future in as_completed(future_to_text):
            text, expected = future_to_text[future]
            try:
                result, request_time = future.result()
                is_question = result['is_question']
                correct = is_question == expected
                if correct:
                    correct_predictions += 1
                total_time += request_time
                results.append((text, expected, is_question, correct, request_time))
                if not correct:
                    incorrect_results.append((text, expected, is_question, request_time))
            except Exception as exc:
                print(f'{text} generated an exception: {exc}')

    accuracy = correct_predictions / len(test_cases) * 100
    avg_response_time = total_time / len(test_cases) * 1000 # Convert to milliseconds

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average response time: {avg_response_time:.2f} ms")

    print("\nDetailed Results:")
    for text, expected, predicted, correct, time in results:
        print(f"Text: {text}")
        print(f"Expected: {'Question' if expected else 'Statement'}")
        print(f"Predicted: {'Question' if predicted else 'Statement'}")
        print(f"Correct: {'Yes' if correct else 'No'}")
        print(f"Response time: {time*1000:.2f} ms")
        print()

    print("\nIncorrect Answers:")
    for text, expected, predicted, time in incorrect_results:
        print(f"Text: {text}")
        print(f"Expected: {'Question' if expected else 'Statement'}")
        print(f"Predicted: {'Question' if predicted else 'Statement'}")
        print(f"Response time: {time*1000:.2f} ms")
        print()

if __name__ == "__main__":
    run_benchmark()
