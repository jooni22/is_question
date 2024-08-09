import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from test_cases_iq import test_cases

API_URL = "http://localhost:11434/api/chat"

def send_request(text):
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gemma2:2b",
        "temperature": 0,
        "max_tokens": 5,
        "messages": [
            {
                "role": "system",
                "content": "You are a specialized language model designed to detect whether a given text sequence is a question or not. Your task is to analyze the input and respond with ONLY \"true\" if the text is a question, or \"false\" if it is not a question. Do not provide any additional explanation or context."
            },
            {
                "role": "user",
                "content": text
            }
        ]
    }
    
    start_time = time.time()
    response = requests.post(API_URL, headers=headers, json=data)
    end_time = time.time()
    
    if response.status_code == 200:
        response_lines = response.text.strip().split('\n')
        last_line = json.loads(response_lines[-1])
        content = last_line['message']['content'].strip().lower()
        is_question = content == 'true'
        return {"is_question": is_question}, end_time - start_time
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None, end_time - start_time

def run_benchmark():
    correct_predictions = 0
    total_time = 0
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_text = {executor.submit(send_request, text): (text, expected) for text, expected in test_cases}
        for future in as_completed(future_to_text):
            text, expected = future_to_text[future]
            try:
                result, request_time = future.result()
                if result is not None:
                    is_question = result['is_question']
                    correct = is_question == expected
                    if correct:
                        correct_predictions += 1
                    total_time += request_time
                    results.append((text, expected, is_question, correct, request_time))
            except Exception as exc:
                print(f'{text} generated an exception: {exc}')

    accuracy = correct_predictions / len(test_cases) * 100
    avg_response_time = total_time / len(test_cases) * 1000  # Convert to milliseconds

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average response time: {avg_response_time:.2f} ms")

    print("\nDetailed Results:")
    for text, expected, predicted, correct, request_time in results:
        print(f"Text: {text}")
        print(f"Expected: {'Question' if expected else 'Statement'}")
        print(f"Predicted: {'Question' if predicted else 'Statement'}")
        print(f"Correct: {'Yes' if correct else 'No'}")
        print(f"Response time: {request_time*1000:.2f} ms")
        print()

if __name__ == "__main__":
    run_benchmark()