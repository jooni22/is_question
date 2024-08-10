import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from main_test_cases_eq import test_cases
import os

# Lista modeli do przetestowania
models = [
    "shahrukhx01/question-vs-statement-classifier",
    "/root/is_question/TEST/fine_tuned_model",
    #"google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
]

# Funkcja do testowania modelu
def test_model(model_name, test_cases):
    if model_name.startswith("./"):
        model_path = os.path.abspath(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    print(f"\nWyniki dla modelu {model_name}:")

    questions_found = 0
    total_cases = len(test_cases)
    results = []

    for i, case in enumerate(test_cases):
        context = case['context']
        question = "Print questions!"
        
        inputs = tokenizer(question, context, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1

        answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])

        results.append(answer)

        print(f"Przypadek {i+1}:")
        print(f"Kontekst: {context}")
        print(f"Znalezione pytanie: {answer}")
        print()

        if answer.lower() != "no question found":
            questions_found += 1

    success_rate = (questions_found / total_cases) * 100

    print(f"Podsumowanie dla modelu {model_name}:")
    print(f"Znalezionych pytań: {questions_found}")
    print(f"Całkowita liczba przypadków: {total_cases}")
    print(f"Procent sukcesu: {success_rate:.2f}%")
    print("=" * 50)

    return questions_found, total_cases, success_rate, results

# Testowanie każdego modelu
overall_results = []
model_answers = []
for model_name in models:
    results = test_model(model_name, test_cases)
    overall_results.append((model_name, *results[:3]))
    model_answers.append(results[3])

# Wypisanie ogólnego podsumowania
print("\nOgólne podsumowanie:")
for model_name, questions_found, total_cases, success_rate in overall_results:
    print(f"{model_name}:")
    print(f"  Znalezionych pytań: {questions_found}")
    print(f"  Całkowita liczba przypadków: {total_cases}")
    print(f"  Procent sukcesu: {success_rate:.2f}%")
    print()

# Znalezienie najlepszego modelu
best_model = max(overall_results, key=lambda x: x[3])
print(f"Najlepszy model: {best_model[0]} z procentem sukcesu {best_model[3]:.2f}%")

# Porównanie znalezionych pytań
print("\nPorównanie znalezionych pytań:")
for i, case in enumerate(test_cases):
    print(f"\nKontekst: {case['context']}")
    for j, model_name in enumerate(models):
        print(f"Znalezione pytanie modelu {j+1}: {model_answers[j][i]}")