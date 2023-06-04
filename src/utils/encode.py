from typing import Hashable


class CategoricalEncoder:
    def __init__(self) -> None:
        self.c2i: dict[Hashable, int] = {}
        self._eval = False

    @property
    def n_categories(self) -> int:
        return len(self.c2i)

    def encode(self, category: Hashable) -> int:
        if category not in self.c2i:
            if self._eval:
                return 0
            self.c2i[category] = len(self.c2i) + 1
        return self.c2i[category]

    def eval(self, set_to: bool = True) -> None:
        self._eval = set_to
