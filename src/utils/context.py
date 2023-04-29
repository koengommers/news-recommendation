class Context:
    def __init__(self):
        self.values = {}

    def add(self, key, value):
        if key in self.values:
            raise Exception(f"Key {key} already in context")
        self.values[key] = value

    def read(self, key, default=None):
        if key not in self.values:
            if default is None:
                raise Exception(f"Key {key} not in context")
            return default
        return self.values[key]


context = Context()
