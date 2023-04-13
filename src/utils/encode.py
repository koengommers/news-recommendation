class CategoricalEncoder:
    def __init__(self) -> None:
        self.c2i: dict[str, int] = {}

    @property
    def n_categories(self) -> int:
        return len(self.c2i)

    def encode(self, category: str) -> int:
        if category not in self.c2i:
            self.c2i[category] = len(self.c2i) + 1
        return self.c2i[category]
