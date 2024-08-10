# Question Detection and Extraction Benchmark

This project benchmarks different approaches to question detection and extraction using various models and techniques.

## Components

1. HuggingFace model for question detection
2. Ollama model for question detection
3. Groq model for question detection
4. spaCy model for question extraction

## Setup

1. Install dependencies:

```
pip install fastapi uvicorn transformers torch spacy requests
python -m spacy download en_core_web_sm
```

2. Download and install Ollama from https://ollama.ai/

3. Set up environment variables:
   - Set `GROQ_API_KEY` for the Groq API

## Running the Benchmarks

### HuggingFace Model

1. Start the HuggingFace model server:

```
python is_question/hf_model_host.py
```

2. Run the benchmark:

```
python is_question/bench_hf.py
```

### Ollama Model

1. Pull the Gemma model:

```
ollama pull gemma2:2b
```

2. Run the benchmark:

```
python is_question/bench_llm_ollama.py
```

### Groq Model

Run the benchmark:

```
python is_question/bench_llm_groq.py
```

### spaCy Model for Question Extraction

1. Start the spaCy model server:

```
python extract_question/spacy_model_host.py
```

2. Run the benchmark:

```
python extract_question/bench_spacy_eq.py
```

## Understanding the Scripts

### Question Detection Scripts

- `is_question/hf_model_host.py`: 
  - Loads a pre-trained HuggingFace model for question detection
  - Sets up a FastAPI server to host the model
  - Defines an endpoint that accepts text input and returns a prediction

- `is_question/bench_hf.py`:
  - Loads test cases from `test_cases_iq.py`
  - Sends each test case to the HuggingFace model server
  - Measures accuracy and response time
  - Outputs detailed results and overall performance metrics

- `is_question/bench_llm_ollama.py`:
  - Uses the Ollama CLI to interact with the Gemma model
  - Processes each test case from `test_cases_iq.py`
  - Measures accuracy and response time
  - Outputs detailed results and overall performance metrics

- `is_question/bench_llm_groq.py`:
  - Uses the Groq API to process test cases
  - Loads test cases from `test_cases_iq.py`
  - Measures accuracy and response time
  - Outputs detailed results and overall performance metrics

### Question Extraction Scripts

- `extract_question/spacy_model_host.py`:
  - Loads the spaCy English language model
  - Sets up a FastAPI server to host the model
  - Defines an endpoint that accepts text input and extracts questions

- `extract_question/bench_spacy_eq.py`:
  - Loads test cases from `test_cases_eq.py`
  - Sends each test case to the spaCy model server
  - Measures accuracy and response time for question extraction
  - Outputs detailed results and overall performance metrics

## Test Cases

- `is_question/test_cases_iq.py`: Contains test cases for question detection
  - Each test case includes input text and expected output (is it a question or not)

- `extract_question/test_cases_eq.py`: Contains test cases for question extraction
  - Each test case includes input text and expected extracted questions

You can modify these files to add or change test cases.

## Results

Each benchmark script will output:
- Overall accuracy
- Average response time
- Detailed results for each test case

Compare these results to evaluate the performance of different approaches to question detection and extraction.

## Customization

- To use different models or datasets, modify the respective script and test case files.
- Adjust hyperparameters or prompts in the benchmark scripts to optimize performance.