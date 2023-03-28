class CategoricalEncoder:
    def __init__(self):
        self.c2i = {}

    @property
    def n_categories(self):
        return len(self.c2i)

    def encode(self, category):
        if category not in self.c2i:
            self.c2i[category] = len(self.c2i) + 1
        return self.c2i[category]
