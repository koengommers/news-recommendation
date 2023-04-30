class Context:
    def __init__(self):
        self.values = {}

    def add(self, key, value):
        if key in self.values:
            raise Exception(f"Key {key} already in context")
        self.values[key] = value

    def read(self, key):
        if key not in self.values:
            raise Exception(f"Key {key} not in context")
        return self.values[key]

    def fill(self, **fill_kwargs):
        def decorator(func):
            def inner(*args, **kwargs):
                final_kwargs = {
                    param: self.read(value)
                    for param, value in fill_kwargs.items()
                    if value in self.values
                }
                final_kwargs.update(kwargs)
                return func(*args, **final_kwargs)

            return inner

        return decorator


context = Context()
