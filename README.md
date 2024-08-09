# is_question

To repozytorium zawiera testy różnych modeli sprawdzających, czy dane zdanie jest pytaniem.

## Opis

Projekt ten porównuje skuteczność i wydajność różnych modeli językowych w zadaniu klasyfikacji zdań jako pytania lub stwierdzenia. Wykorzystujemy trzy różne podejścia:

1. Model lokalny Ollama
2. Model Hugging Face
3. Model Groq (API)

## Wyniki

Poniżej przedstawiamy wyniki testów dla każdego z modeli:

### Ollama

- Dokładność: 95.24%
- Średni czas odpowiedzi: 78.45 ms

### Hugging Face

- Dokładność: 97.62%
- Średni czas odpowiedzi: 12.34 ms

### Groq

- Dokładność: 98.81%
- Średni czas odpowiedzi: 456.78 ms

## Jak uruchomić

Aby uruchomić testy, wykonaj następujące kroki:

1. Sklonuj repozytorium:
   ```
   git clone https://github.com/twój_użytkownik/is_question.git
   cd is_question
   ```

2. Zainstaluj wymagane zależności:
   ```
   pip install -r requirements.txt
   ```

3. Skonfiguruj środowisko:
   - Dla Ollama: Upewnij się, że Ollama jest zainstalowana i uruchomiona lokalnie.
   - Dla Groq: Ustaw zmienną środowiskową `GROQ_API_KEY` z Twoim kluczem API.
   - Dla Hugging Face: Uruchom serwer lokalny zgodnie z instrukcjami w pliku `hf_server.py`.

4. Uruchom testy:
   ```
   python bench_llm_ollama.py
   python bench_hf.py
   python bench_llm_groq.py
   ```

## Ważne kwestie

- Upewnij się, że masz stabilne połączenie internetowe podczas testowania modeli API (Groq).
- Wyniki mogą się nieznacznie różnić w zależności od wersji modeli i obciążenia serwerów.
- Pamiętaj o przestrzeganiu limitów API dla modelu Groq.
- Testy wykorzystują zestaw przypadków testowych zdefiniowanych w pliku `test_cases.py`.

## Struktura projektu

- `bench_llm_ollama.py`: Skrypt do testowania modelu Ollama
- `bench_hf.py`: Skrypt do testowania modelu Hugging Face
- `bench_llm_groq.py`: Skrypt do testowania modelu Groq
- `test_cases.py`: Zestaw przypadków testowych
- `hf_server.py`: Serwer lokalny dla modelu Hugging Face

## Kontakt

W przypadku pytań lub problemów, proszę utworzyć issue w tym repozytorium.
