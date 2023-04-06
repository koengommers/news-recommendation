from nltk.tokenize import word_tokenize


class NltkTokenizer:
    def __init__(self):
        self.t2i = {}
        self._eval = False

    @property
    def vocab_size(self):
        return len(self.t2i)

    def token2int(self, token):
        if token not in self.t2i:
            if self._eval:
                return 0
            self.t2i[token] = len(self.t2i) + 1
        return self.t2i[token]

    def eval(self, set_to=True):
        self._eval = set_to

    def __call__(self, text, length):
        tokens = word_tokenize(text)
        tokens = [self.token2int(token) for token in tokens]
        if len(tokens) < length:
            padding_length = length - len(tokens)
            return tokens + [0] * padding_length
        else:
            return tokens[:length]
