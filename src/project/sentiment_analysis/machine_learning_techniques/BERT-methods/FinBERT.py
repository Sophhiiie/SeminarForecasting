# This class computes the sentiment using hugging face

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load FinBERT from Hugging Face
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a sentiment analysis pipeline
finbert_sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Sample financial text
text = "Tesla stock is performing incredibly well this quarter."

# Run sentiment analysis
result = finbert_sentiment(text)
print("FinBERT result:", result)
