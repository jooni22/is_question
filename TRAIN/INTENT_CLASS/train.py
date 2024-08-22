import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizerFast, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    get_cosine_schedule_with_warmup,
    EarlyStoppingCallback
)
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Load and prepare data
train_df = pd.read_csv("TRAIN/INTENT_CLASS/translated_kor_3i4k_csv/train_fix_v1.csv")
test_df = pd.read_csv("TRAIN/INTENT_CLASS/translated_kor_3i4k_csv/test_fix_v1.csv")

# Convert to Dataset objects
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer and model from the fine-tuned directory
model_path = "TRAIN/INTENT_CLASS/further_fine_tuned_model"
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

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

# Adjust training arguments for continued training
training_args = TrainingArguments(
    output_dir="TRAIN/INTENT_CLASS/further_fine_tuned_model_2_fixDS/results_continued",
    num_train_epochs=2,  # Increased the number of epochs
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=200,
    weight_decay=0.1,
    logging_dir='./logs_continued',
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    learning_rate=1e-5,
)

# # Adjust training arguments for continued training
# training_args = TrainingArguments(
#     output_dir="./results_continued",
#     num_train_epochs=1,  # Increase the number of epochs
#     per_device_train_batch_size=64,  # Reduce batch size to allow for larger learning rate
#     per_device_eval_batch_size=64,
#     warmup_steps=1000,  # Reduce warmup steps
#     weight_decay=0.02,  # Slightly increase weight decay
#     logging_dir='./logs_continued',
#     logging_steps=100,
#     evaluation_strategy="steps",  # Evaluate more frequently
#     eval_steps=1000,  # Evaluate every 100 steps
#     save_strategy="steps",
#     save_steps=1000,
#     load_best_model_at_end=True,
#     learning_rate=2e-5,  # Slightly lower learning rate for fine-tuning
#     metric_for_best_model="f1",  # Use F1 score to determine the best model
# )

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Added Early Stopping
)

# Calculate the total number of training steps
total_steps = len(tokenized_train) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs

# Set the learning rate schedule
trainer.create_optimizer_and_scheduler(num_training_steps=total_steps)
trainer.optimizer.learning_rate = get_cosine_schedule_with_warmup(
    trainer.optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=total_steps
)

# Continue training the model
trainer.train()

# Save the further fine-tuned model
model.save_pretrained("TRAIN/INTENT_CLASS/further_fine_tuned_model_2_fixDS")
tokenizer.save_pretrained("TRAIN/INTENT_CLASS/further_fine_tuned_model_2_fixDS")

print("Continued training completed. Model saved in TRAIN/INTENT_CLASS/further_fine_tuned_model_2_fixDS")

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