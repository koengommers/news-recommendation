from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer


class NltkTokenizer:
    def __init__(self) -> None:
        self.t2i: dict[str, int] = {}
        self._eval = False

    @property
    def vocab_size(self) -> int:
        return len(self.t2i)

    def token2int(self, token: str) -> int:
        if token not in self.t2i:
            if self._eval:
                return 0
            self.t2i[token] = len(self.t2i) + 1
        return self.t2i[token]

    def eval(self, set_to: bool = True) -> None:
        self._eval = set_to

    def __call__(self, text: str, length: int) -> list[int]:
        tokens: list[str] = word_tokenize(text)
        ints = [self.token2int(token) for token in tokens]
        if len(ints) < length:
            padding_length = length - len(ints)
            return ints + [0] * padding_length
        else:
            return ints[:length]


class BertTokenizer:
    def __init__(self, pretrained_model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.pretrained_model_name = pretrained_model_name

    def eval(self):
        pass

    def __call__(self, text: str, length: int) -> dict[str, list[int]]:
        return dict(
            self.tokenizer(
                text, max_length=length, padding="max_length", truncation=True
            )
        )
