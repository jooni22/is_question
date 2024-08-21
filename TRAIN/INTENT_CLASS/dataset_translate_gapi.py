from datasets import load_dataset
from googletrans import Translator
from tqdm import tqdm

# Załaduj dataset
dataset = load_dataset("wicho/kor_3i4k")

# Inicjalizuj translator
translator = Translator()

def translate_text(text):
    try:
        return translator.translate(text, src='ko', dest='en').text
    except:
        return text  # W przypadku błędu, zwróć oryginalny tekst

def translate_example(example):
    example['text'] = translate_text(example['text'])
    return example

# Przetłumacz dataset
translated_dataset = dataset.map(translate_example, batched=False, desc="Tłumaczenie datasetu")

# Zapisz przetłumaczony dataset
translated_dataset.save_to_disk("translated_kor_3i4k")

# Wydrukuj przykład
print(translated_dataset['train'][0])