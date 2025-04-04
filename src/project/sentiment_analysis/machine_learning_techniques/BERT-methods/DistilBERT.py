from .base_bert import BaseBERTSentiment

class DistilBERT(BaseBERTSentiment):
    def __init__(self):
        super().__init__("distilbert-base-uncased-finetuned-sst-2-english")

