from .base_bert import BaseBERTSentiment

class TweetBERT(BaseBERTSentiment):
    def __init__(self):
        super().__init__("finiteautomata/bertweet-base-sentiment-analysis")


