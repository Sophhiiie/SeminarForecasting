from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Use a fine-tuned BERTweet for sentiment
model_name = "finiteautomata/bertweet-base-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create sentiment pipeline
tweetbert_sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Sample tweet
tweet = "I'm so hyped about NVIDIA's earnings! 🚀"

# Run sentiment
result = tweetbert_sentiment(tweet)
print("TweetBERT result:", result)
