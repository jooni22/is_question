import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Test data
test_cases = [
    ("Is this a question?", True),
    ("This is a statement.", False),
    ("This is a not statement.", False),    
    ("What time is it?", True),
    ("I'm not sure if this is a question.", False),
    ("Can you help me?", True),
    ("The sky is blue.", False),
    ("How does this work?", True),
    ("It's raining outside.", False),
    ("Do you know the way to San Jose?", True),
    ("I like ice cream.", False),
    ("Why is the sky blue?", True),
    ("Life is what happens when you're busy making other plans.", False),
    ("Have you ever seen the rain?", True),
    ("To be or not to be, that is the question.", True),
    ("I wonder if this is a question.", True),
    ("Could you please pass the salt?", True),
    ("The early bird catches the worm.", False),
    ("Who let the dogs out?", True),
    ("Actions speak louder than words.", False),
    ("Isn't it a beautiful day?", True),
    ("That's a rhetorical question, isn't it?", True),
    ("She asked if I could come to the party.", False),
    ("How about we go for a walk?", True),
    ("I'd like to know more about this topic.", True),
    ("What if I told you the Earth was flat?", True),
    ("It's not rocket science.", False),
    ("Are we there yet?", True),
    ("The question is, will it work?", True),
    ("I'm telling you, not asking.", False),
    ("You call that a question?", False),
    ("This statement ends with a question mark?", True),
    ("Just because it ends with a period doesn't mean it's not a question.", False),
    ("Why bother?", True),
    ("I'm Ron Burgundy?", True),
    ("Do fish sleep?", True),
    ("The answer, my friend, is blowin' in the wind.", False),
    ("How much wood would a woodchuck chuck if a woodchuck could chuck wood?", True),
    ("I have a dream.", False),
    ("Et tu, Brute?", True),
    ("This is not a pipe.", False),
    ("Hi, how are you? I hope everything's okay with you. I heard you were on vacation recently. Where exactly did you go? It must have been great! And what did you enjoy most about the trip?", True),
    ("The weather has been quite unpredictable lately. One day it's sunny, the next it's raining. Do you think climate change is affecting our local weather patterns?", True),
    ("I've been thinking about changing careers. It's not an easy decision to make. Have you ever considered a major life change like that?", True),
    ("The new restaurant downtown is getting rave reviews. I heard their chef trained in Paris. We should try it sometime, don't you think?", False),
    ("Life is full of surprises. Sometimes good, sometimes bad. But isn't that what makes it interesting?", False),
    ("I'm working on a new project at work. It's challenging but rewarding. I'm learning a lot of new skills in the process.", False),
    ("The book you recommended was fantastic. I couldn't put it down. The author's writing style was so engaging.", False),
    ("Can you believe how fast technology is advancing? It seems like there's a new gadget every week. What do you think the next big innovation will be?", True),
    ("I've been trying to eat healthier lately. It's not easy with all the temptations around. Do you have any tips for sticking to a healthy diet?", True),
    ("The concert last night was amazing. The band played all their hits. The crowd was so energetic. I'm still buzzing from the experience.", False),
    ("Have you heard about the new environmental initiative in our city? They're planting trees all over. It's supposed to help reduce carbon emissions. What are your thoughts on this approach?", True),
    ("I'm planning a surprise party for my best friend. It's been quite stressful keeping it a secret. Do you think she'll like it? I hope everything goes smoothly.", True),
    ("The history of our town is fascinating. Did you know it was founded over 200 years ago? There are so many interesting stories from the past.", False),
    ("I've been learning a new language online. It's challenging but fun. They say immersion is the best way to learn. Maybe I should plan a trip to practice?", False),
    ("The new art exhibition at the museum is thought-provoking. It challenges our perceptions of reality. The artist uses unconventional materials to create stunning pieces.", False),
    ("Why do we drive on parkways and park on driveways? Isn't language funny sometimes? These are the kinds of questions that keep me up at night.", True),
    ("I'm considering adopting a pet. It's a big responsibility. Do you have any experience with pets? What kind would you recommend for a first-time owner?", True),
    ("The documentary about space exploration was mind-blowing. It showed how far we've come in understanding the universe. Yet there's still so much we don't know.", False),
    ("Have you ever wondered why the sky is blue? Or why we dream? These fundamental questions about our world are still being researched by scientists.", True),
    ("The new coffee shop on the corner makes the best lattes. Their pastries are delicious too. We should meet there for our next catch-up.", False),
]



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
