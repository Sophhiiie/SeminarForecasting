from .base_bert import BaseBERTSentiment

class FinBERT(BaseBERTSentiment):
    def __init__(self):
        super().__init__("yiyanghkust/finbert-tone")

