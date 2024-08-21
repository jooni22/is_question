import pandas as pd
import numpy as np
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytaj model i tokenizer
model_path = "TRAIN/INTENT_CLASS/fine_tuned_model"
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Wczytaj dane testowe
test_df = pd.read_csv("TRAIN/INTENT_CLASS/translated_kor_3i4k_csv/test.csv")

# Tokenizacja danych testowych
encoded_data = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, return_tensors="pt")

# Predykcja
with torch.no_grad():
    outputs = model(**encoded_data)
    predictions = torch.argmax(outputs.logits, dim=-1)

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

print(f"Dokładność: {accuracy:.4f}")
print(f"Precyzja: {precision:.4f}")
print(f"Czułość: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Szczegółowy raport klasyfikacji
print("\nRaport klasyfikacji:")
print(classification_report(true_labels, predictions, target_names=model.config.id2label.values()))

# Analiza błędnie sklasyfikowanych przykładów
misclassified = test_df[predictions != true_labels]
print("\nPrzykłady błędnie sklasyfikowane:")
for _, row in misclassified.iterrows():
    print(f"Tekst: {row['text']}")
    print(f"Prawdziwa etykieta: {model.config.id2label[row['label']]}")
    print(f"Przewidziana etykieta: {model.config.id2label[predictions[_]]}")
    print("---")

# Rozkład predykcji dla poszczególnych klas
plt.figure(figsize=(10, 6))
sns.countplot(x=predictions)
plt.title('Rozkład predykcji dla poszczególnych klas')
plt.xlabel('Klasy')
plt.ylabel('Liczba predykcji')
plt.savefig('TRAIN/INTENT_CLASS/prediction_distribution.png')
plt.close()

print("\nAnalizy zostały zakończone. Wykresy zostały zapisane w plikach 'confusion_matrix.png' i 'prediction_distribution.png'.")