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

# Load and prepare data
train_df = pd.read_csv("/root/is_question/TRAIN/INTENT_CLASS/translated_kor_3i4k_csv/train_fix_v1.csv")
test_df = pd.read_csv("/root/is_question/TRAIN/INTENT_CLASS/translated_kor_3i4k_csv/test_fix_v1.csv")

# Convert to Dataset objects
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Initialize tokenizer and model
model_name = "FacebookAI/roberta-base"
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
    output_dir="/root/is_question/TRAIN/INTENT_CLASS/roberta_base_scratch",
    num_train_epochs=1,  # Ustawione na 10, ale z early stopping
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    learning_rate=5e-05,
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
model.save_pretrained("/root/is_question/TRAIN/INTENT_CLASS/roberta_base_scratch")
tokenizer.save_pretrained("/root/is_question/TRAIN/INTENT_CLASS/roberta_base_scratch")

print("Training completed. Model saved.")

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
test_df.to_csv("/root/is_question/TRAIN/INTENT_CLASS/roberta_base_scratch/test_predictions.csv", index=False)
print("Predictions saved")