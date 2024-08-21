from datasets import load_dataset
import pandas as pd
import os

# Załaduj dataset
print("Ładowanie datasetu...")
dataset = load_dataset("wicho/kor_3i4k")

# Zapisz dataset na dysku
print("Zapisywanie datasetu na dysku...")
dataset.save_to_disk("kor_3i4k_dataset")

# Konwertuj do CSV
print("Konwertowanie do CSV...")

# Utwórz folder na pliki CSV, jeśli nie istnieje
os.makedirs("kor_3i4k_csv", exist_ok=True)

# Funkcja do podziału i zapisu dataframe'ów
def split_and_save_csv(df, split_name):
    # Grupuj według etykiety
    grouped = df.groupby('label')
    
    # Dla każdej grupy, zapisz osobny plik CSV
    for label, group in grouped:
        csv_path = f"kor_3i4k_csv/{label}_{split_name}.csv"
        group.to_csv(csv_path, index=False)
        print(f"Zapisano {csv_path}")

# Konwertuj każdy split (train, test) do CSV
for split in dataset.keys():
    df = pd.DataFrame(dataset[split])
    split_and_save_csv(df, split)

print("Konwersja zakończona.")

# Wyświetl pierwsze kilka wierszy z pliku CSV dla sprawdzenia
print("\nPierwsze 5 wierszy z pliku 0_train.csv:")
print(pd.read_csv("kor_3i4k_csv/0_train.csv").head())

# Policz liczbę unikalnych etykiet
unique_labels = set()
for split in dataset.keys():
    unique_labels.update(dataset[split]['label'])

print(f"\nLiczba unikalnych etykiet: {len(unique_labels)}")
print(f"Unikalne etykiety: {sorted(unique_labels)}")