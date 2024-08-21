import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Load and prepare data
train_df = pd.read_csv("TRAIN/INTENT_CLASS/translated_kor_3i4k_csv/train.csv")
test_df = pd.read_csv("TRAIN/INTENT_CLASS/translated_kor_3i4k_csv/test.csv")

# Convert to Dataset objects
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer and model
model_name = "bespin-global/klue-roberta-small-3i4k-intent-classification"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

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
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Training completed. Model saved in ./fine_tuned_model")

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
test_df.to_csv("TRAIN/INTENT_CLASS/test_predictions.csv", index=False)
print("Predictions saved to TRAIN/INTENT_CLASS/test_predictions.csv")