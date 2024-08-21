from datasets import load_from_disk
import pandas as pd
import os

# Wczytaj dataset
dataset = load_from_disk("/root/is_question/translated_kor_3i4k")

# Utwórz folder na pliki CSV, jeśli nie istnieje
output_dir = "/root/is_question/translated_kor_3i4k_csv"
os.makedirs(output_dir, exist_ok=True)

# Konwertuj i zapisz każdy split jako osobny plik CSV
for split in dataset.keys():
    df = pd.DataFrame(dataset[split])
    csv_path = os.path.join(output_dir, f"{split}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Zapisano {split} do {csv_path}")

print("Konwersja zakończona.")