import re
from transformers import AutoTokenizer


# Strategy: Definisce un'interfaccia per la tokenizzazione
class TokenizerStrategy:
    def tokenize(self, text):
        raise NotImplementedError("Tokenize method must be implemented")


# Concrete Strategy 1: Tokenizzazione senza modello pre-addestrato
class SimpleTokenizer(TokenizerStrategy):
    def tokenize(self, text):
        text = re.sub(r'Lezione \d+', '', text)
        text = re.sub(r'', '', text)
        text = re.sub(r'\d{2}/\d{2}/\d{4}', '', text)
        return re.findall(r'\S+', text)


# Concrete Strategy 2: Tokenizzazione con modello pre-addestrato
class PretrainedTokenizer(TokenizerStrategy):
    def __init__(self, model):
        if not model:
            raise ValueError("Un modello pre-addestrato è richiesto per la tokenizzazione.")
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)


# Factory per la creazione di tokenizer
class TokenizerFactory:
    @staticmethod
    def create_tokenizer(model=None):
        if model is None:
            return SimpleTokenizer()
        else:
            return PretrainedTokenizer(model)



