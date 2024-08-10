import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from is_question.test_cases_iq import test_cases



API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.environ.get("GROQ_API_KEY")

def send_request(texts):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    prompt = "\n".join([f'"{text}"' for text in texts])
    data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a specialized language model designed to detect whether given text sequences are questions or not. Your task is to analyze the input and respond with ONLY \"true\" if the text is a question, or \"false\" if it is not a question. Provide your answers as a space-separated list of true/false values, one for each input text. Do not provide any additional explanation or context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": "gemma2-9b-it",
        "temperature": 0,
        "max_tokens": 50,
        "top_p": 1,
        "stream": False,
        "stop": None
    }
    
    start_time = time.time()
    response = requests.post(API_URL, headers=headers, json=data)
    end_time = time.time()
    
    if response.status_code == 200:
        result = response.json()
        predictions = result['choices'][0]['message']['content'].strip().split()
        return [p.lower() == 'true' for p in predictions], end_time - start_time
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None, end_time - start_time

def run_benchmark():
    correct_predictions = 0
    total_time = 0
    results = []
    incorrect_results = []
    
    # Group test cases into batches of 5
    batches = [test_cases[i:i+5] for i in range(0, len(test_cases), 5)]
    
    for batch in batches:
        texts, expected = zip(*batch)
        try:
            predictions, request_time = send_request(texts)
            if predictions is not None:
                for text, exp, pred in zip(texts, expected, predictions):
                    correct = pred == exp
                    if correct:
                        correct_predictions += 1
                    results.append((text, exp, pred, correct, request_time / len(batch)))
                    if not correct:
                        incorrect_results.append((text, exp, pred, request_time / len(batch)))
                total_time += request_time
            # Add 1-second delay between batches
            time.sleep(1)
        except Exception as exc:
            print(f'Batch generated an exception: {exc}')
    
    accuracy = correct_predictions / len(test_cases) * 100
    avg_response_time = total_time / len(test_cases) * 1000 # Convert to milliseconds
    
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
    
    print("\nIncorrect Answers:")
    for text, expected, predicted, request_time in incorrect_results:
        print(f"Text: {text}")
        print(f"Expected: {'Question' if expected else 'Statement'}")
        print(f"Predicted: {'Question' if predicted else 'Statement'}")
        print(f"Response time: {request_time*1000:.2f} ms")
        print()

if __name__ == "__main__":
    run_benchmark()
