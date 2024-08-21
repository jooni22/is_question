import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import resource
import sys

# Ograniczenie pamięci do 1GB
def limit_memory(max_memory):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory, hard))

# Ograniczenie pamięci do 1GB
max_memory = 10 * 1024 * 1024 * 1024  # 1GB w bajtach
limit_memory(max_memory)

try:
    # Wczytaj model i tokenizer
    model_path = "TRAIN/INTENT_CLASS/fine_tuned_model"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)

    # Wczytaj dane testowe
    test_df = pd.read_csv("TRAIN/INTENT_CLASS/translated_kor_3i4k_csv/test.csv")

    # Tokenizacja danych testowych
    encoded_data = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, return_tensors="pt")

    # Predykcja
    predictions = []
    batch_size = 32
    for i in range(0, len(test_df), batch_size):
        batch = {k: v[i:i+batch_size] for k, v in encoded_data.items()}
        with torch.no_grad():
            outputs = model(**batch)
            predictions.extend(torch.argmax(outputs.logits, dim=-1).tolist())

    # Konwersja etykiet na liczby całkowite
    true_labels = test_df['label'].astype(int)

    # Obliczenie macierzy pomyłek
    cm = confusion_matrix(true_labels, predictions)

    # Wizualizacja macierzy pomyłek
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Macierz pomyłek')
    plt.ylabel('Prawdziwe etykiety')
    plt.xlabel('Przewidziane etykiety')
    plt.savefig('TRAIN/INTENT_CLASS/confusion_matrix.png')
    plt.close()

    # Obliczenie metryk
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')

    # Zapisywanie wyników do pliku
    with open('TRAIN/INTENT_CLASS/results.txt', 'w') as f:
        f.write(f"Dokładność: {accuracy:.4f}\n")
        f.write(f"Precyzja: {precision:.4f}\n")
        f.write(f"Czułość: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write("\nRaport klasyfikacji:\n")
        f.write(classification_report(true_labels, predictions, target_names=model.config.id2label.values()))

    # Analiza błędnie sklasyfikowanych przykładów
    misclassified = test_df[np.array(predictions) != true_labels]
    with open('TRAIN/INTENT_CLASS/misclassified_examples.txt', 'w') as f:
        f.write("\nPrzykłady błędnie sklasyfikowane:\n")
        for _, row in misclassified.iterrows():
            f.write(f"Tekst: {row['text']}\n")
            f.write(f"Prawdziwa etykieta: {model.config.id2label[row['label']]}\n")
            f.write(f"Przewidziana etykieta: {model.config.id2label[predictions[_]]}\n")
            f.write("---\n")

    # Rozkład predykcji dla poszczególnych klas
    plt.figure(figsize=(10, 6))
    sns.countplot(x=predictions)
    plt.title('Rozkład predykcji dla poszczególnych klas')
    plt.xlabel('Klasy')
    plt.ylabel('Liczba predykcji')
    plt.savefig('TRAIN/INTENT_CLASS/prediction_distribution.png')
    plt.close()

    print("\nAnalizy zostały zakończone. Wyniki zostały zapisane w plikach 'results.txt', 'misclassified_examples.txt', 'confusion_matrix.png' i 'prediction_distribution.png'.")

except MemoryError:
    sys.stderr.write('\n\nERROR: Przekroczono limit pamięci\n')
    sys.exit(1)