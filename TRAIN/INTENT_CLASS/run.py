from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TextClassificationPipeline

# Load fine-tuned model by HuggingFace Model Hub
HUGGINGFACE_MODEL_PATH = "bespin-global/klue-roberta-small-3i4k-intent-classification"
loaded_tokenizer = RobertaTokenizerFast.from_pretrained(HUGGINGFACE_MODEL_PATH )
loaded_model = RobertaForSequenceClassification.from_pretrained(HUGGINGFACE_MODEL_PATH )

# using Pipeline
text_classifier = TextClassificationPipeline(
    tokenizer=loaded_tokenizer, 
    model=loaded_model, 
    top_k=None  # This replaces return_all_scores=True
)

# predict
text = "who are you?"

preds_list = text_classifier(text)
best_pred = preds_list[0][0]  # Access the first prediction of the first (and only) input
print(f"Label of Best Intention: {best_pred['label']}")
print(f"Score of Best Intention: {best_pred['score']}")