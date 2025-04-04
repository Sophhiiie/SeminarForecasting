from transformers import pipeline

class BaseBERTSentiment:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pipeline = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

    def analyze(self, text: str):
        return self.pipeline(text)