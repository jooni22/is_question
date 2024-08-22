import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TextClassificationPipeline
from test_case_IC_iq import test_cases
import os
from datetime import datetime

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load fine-tuned model from local path
MODEL_PATH = "TRAIN/INTENT_CLASS/further_fine_tuned_model"
loaded_tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH)
loaded_model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

# Using Pipeline
text_classifier = TextClassificationPipeline(
    tokenizer=loaded_tokenizer, 
    model=loaded_model,
    device=0 if device == "cuda" else -1,  # Use GPU if available
    top_k=None  # This returns all scores
)

# Function to classify a single text
def classify_text(text):
    preds_list = text_classifier(text)
    best_pred = preds_list[0][0]  # Access the first prediction of the first (and only) input
    
    # Treat "question", "command", "rhetorical question", and "rhetorical command" as "question"
    if best_pred['label'] in ["question", "command", "rhetorical question", "rhetorical command"]:
        return "question", best_pred['score']
    else:
        return best_pred['label'], best_pred['score']

# Process test cases
correct_predictions = 0
total_cases = len(test_cases)

# Prepare results string
results = []

for text, expected_is_question in test_cases:
    label, score = classify_text(text)
    predicted_is_question = label == "question"
    
    is_correct = predicted_is_question == expected_is_question
    if is_correct:
        correct_predictions += 1
    
    result = f"Text: {text}\n"
    result += f"Predicted: {'Question' if predicted_is_question else 'Not a question'} (Label: {label}, Score: {score:.4f})\n"
    result += f"Expected: {'Question' if expected_is_question else 'Not a question'}\n"
    result += f"Correct: {'Yes' if is_correct else 'No'}\n"
    result += "-" * 50 + "\n"
    
    results.append(result)
    print(result)

# Calculate accuracy
accuracy = correct_predictions / total_cases
print(f"\nAccuracy: {accuracy:.2%}")

# Create directory if it doesn't exist
os.makedirs("TRAIN/INTENT_CLASS/test_case_results", exist_ok=True)

# Generate filename with current date, time, and accuracy
current_time = datetime.now().strftime("%d%m%Y_%H_%M_%S")
filename = f"{current_time}_{accuracy:.2f}.log"
filepath = os.path.join("TRAIN/INTENT_CLASS/test_case_results", filename)

# Write results to file
with open(filepath, "w") as f:
    f.write("\n".join(results))
    f.write(f"\nAccuracy: {accuracy:.2%}\n")

print(f"Results saved to {filepath}")