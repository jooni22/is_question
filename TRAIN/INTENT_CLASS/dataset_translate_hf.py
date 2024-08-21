import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# Załaduj model i tokenizer
model_name = "Helsinki-NLP/opus-mt-ko-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Przenieś model na GPU, jeśli jest dostępne
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_file(input_file, output_file):
    df = pd.read_csv(input_file)
    df['translated_text'] = df['text'].progress_apply(translate_text)
    df.to_csv(output_file, index=False)
    print(f"Zapisano przetłumaczony plik: {output_file}")

# Utwórz folder na przetłumaczone pliki
output_folder = "TRAIN/INTENT_CLASS/kor_3i4k_csv/translated_hf/"
os.makedirs(output_folder, exist_ok=True)

# Znajdź wszystkie pliki CSV w folderze wejściowym
input_folder = "TRAIN/INTENT_CLASS/kor_3i4k_csv/"
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Posortuj pliki tak, aby pliki test były na początku
csv_files.sort(key=lambda x: (not x.endswith('_train.csv'), x))

# Przetłumacz każdy plik
for file in csv_files:
    input_file = os.path.join(input_folder, file)
    output_file = os.path.join(output_folder, f"translated_{file}")
    
    print(f"Tłumaczenie pliku: {file}")
    tqdm.pandas(desc="Tłumaczenie")
    translate_file(input_file, output_file)

print("Tłumaczenie zakończone.")