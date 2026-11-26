from cryptocast.sentiment.vader import SentimentResult, VaderSentimentScorer


class FinBertSentimentScorer:
    """FinBERT sentiment analyzer with automatic fallback to Vader on CPU/import constraints."""

    def __init__(self, force_fallback: bool = False) -> None:
        self.fallback_scorer = VaderSentimentScorer()
        self.use_finbert = False
        self.model = None
        self.tokenizer = None

        if not force_fallback:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                model_name = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.eval()
                self.torch = torch
                self.use_finbert = True
            except Exception:
                # Graceful fallback to lightweight VADER
                self.use_finbert = False

    def score_text(self, text: str) -> SentimentResult:
        if not self.use_finbert or self.model is None or self.tokenizer is None:
            return self.fallback_scorer.score_text(text)

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                probs = self.torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()

            # FinBERT label order: [positive, negative, neutral]
            pos, neg, neu = probs[0], probs[1], probs[2]
            score = pos - neg
            return SentimentResult(
                pos=round(pos, 4),
                neg=round(neg, 4),
                neu=round(neu, 4),
                score=round(score, 4),
            )
        except Exception:
            return self.fallback_scorer.score_text(text)
