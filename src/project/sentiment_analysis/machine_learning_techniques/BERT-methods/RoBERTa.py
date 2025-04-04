from .base_bert import BaseBERTSentiment

class RoBERTa(BaseBERTSentiment):
    def __init__(self):
        super().__init__("cardiffnlp/twitter-roberta-base-sentiment")

