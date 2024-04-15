import re
import pickle
from collections import Counter

class Tokenizer:
    def __init__(self):
        self.vocab = {}
    
    def tokenize(self, text):
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)  # remove special characters
        tokens = text.lower().split()  # lowercase and split by whitespace
        return tokens

    def build_vocab(self, tokenized_texts):
        word_counts = Counter()
        for text in tokenized_texts:
            word_counts.update(text)
        self.vocab = {word: index + 1 for index, (word, count) in enumerate(word_counts.items())}

    def save_vocab(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)