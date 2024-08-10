import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
from torch.nn import functional as F
from train_data import train_data  # Importujemy train_data z pliku train_data.py

# Funkcja do konwersji odpowiedzi na listę
def ensure_answer_is_list(answer):
    if isinstance(answer, list):
        return answer
    elif isinstance(answer, str):
        return [answer] if answer else []
    else:
        return []

# Przygotowanie danych treningowych
train_dataset = Dataset.from_dict({
    "context": [item["context"] for item in train_data],
    "question": [item["question"] for item in train_data],
    "answer": [ensure_answer_is_list(item["answer"]) for item in train_data]
})

# Inicjalizacja modelu i tokenizera
model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Funkcja do tokenizacji danych
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["question"], examples["context"], truncation=True, padding="max_length", max_length=512, return_offsets_mapping=True)
    
    start_positions = []
    end_positions = []
    
    for i, (offsets, context, answers) in enumerate(zip(tokenized_inputs["offset_mapping"], examples["context"], examples["answer"])):
        answer = answers[0] if answers else ""  # Bierzemy pierwsze pytanie z listy, jeśli jest pusta, używamy pustego stringa
        start_char = context.find(answer)
        end_char = start_char + len(answer)
        
        sequence_ids = tokenized_inputs.sequence_ids(i)
        
        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offsets[context_start][0] > start_char or offsets[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    
    tokenized_inputs["start_positions"] = start_positions
    tokenized_inputs["end_positions"] = end_positions
    return tokenized_inputs

tokenized_dataset = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=train_dataset.column_names)

# Niestandardowa klasa trenera
class QuestionAnsweringTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        start_positions = inputs["start_positions"]
        end_positions = inputs["end_positions"]
        
        start_loss = F.cross_entropy(start_logits, start_positions)
        end_loss = F.cross_entropy(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        
        return (total_loss, outputs) if return_outputs else total_loss

# Konfiguracja treningu
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Trening modelu
trainer = QuestionAnsweringTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Zapisz dostrojony model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")