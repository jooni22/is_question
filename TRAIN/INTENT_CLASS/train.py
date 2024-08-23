import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizerFast, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tensorflow.python.keras.optimizer_v2.adam import Adam
import os
from tqdm import tqdm

# Define the model save path
MODEL_SAVE_PATH = "./custom-dst-roberta-base"

# Ensure the directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

splits = {'train': 'train_fix_v1.csv', 'test': 'test_fix_v1.csv'}
train_df = pd.read_csv("hf://datasets/jooni22/dst-question/" + splits["train"])
test_df = pd.read_csv("hf://datasets/jooni22/dst-question/" + splits["test"])
# # Load and prepare data
# train_df = pd.read_csv("./train_fix_v1.csv")
# test_df = pd.read_csv("./test_fix_v1.csv")

# Convert to Dataset objects
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Initialize tokenizer and model
model_name = "jooni22/custom-dst-roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=7,
    id2label={
        0: "fragment",
        1: "statement",
        2: "question",
        3: "command",
        4: "rhetorical question",
        5: "rhetorical command",
        6: "intonation-dependent utterance"
    },
    label2id={
        "fragment": 0,
        "statement": 1,
        "question": 2,
        "command": 3,
        "rhetorical question": 4,
        "rhetorical command": 5,
        "intonation-dependent utterance": 6
    }
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Compute metrics function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    num_train_epochs=30,
    per_device_train_batch_size=164,  # Increased batch size
    per_device_eval_batch_size=164,    # Increased batch size
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=os.path.join(MODEL_SAVE_PATH, 'logs'),
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    learning_rate=5e-05,
    fp16=True,  # Enabled mixed precision training
    optim="adamw_torch",  # Use AdamW with 8-bit quantization
    dataloader_pin_memory=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print(f"Training completed. Model saved to {MODEL_SAVE_PATH}")

# Evaluate the model on the test set
print("Evaluating model on test set...")
test_results = trainer.evaluate(tokenized_test)

print("Test set evaluation results:")
for key, value in test_results.items():
    print(f"{key}: {value}")

# Perform predictions on the test set
test_predictions = trainer.predict(tokenized_test)

# Get predicted labels
predicted_labels = np.argmax(test_predictions.predictions, axis=1)
true_labels = test_predictions.label_ids

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, 
                            target_names=model.config.id2label.values()))

# Optional: Save predictions to CSV
test_df['predicted_label'] = predicted_labels
predictions_path = os.path.join(MODEL_SAVE_PATH, "test_predictions.csv")
test_df.to_csv(predictions_path, index=False)
print(f"Predictions saved to {predictions_path}")