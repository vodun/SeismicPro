from contextlib import contextmanager


_DEFAULT_CONFIG = {
    "enable_fast_pickling": False,
}


class Config:
    def __init__(self, options):
        self.default_options = options.copy()
        self.options = options.copy()

    def __getitem__(self, key):
        return self.options[key]

    def __setitem__(self, key, value):
        self.options[key] = value

    def reset_options(self, *args):
        for arg in args:
            if arg in self.default_options:
                self.options[arg] = self.default_options[arg]
            else:
                _ = self.options.pop(arg, None)

    @contextmanager
    def use_options(self, **kwargs):
        self.options.update(kwargs)
        yield
        self.reset_options(*kwargs.keys())


config = Config(_DEFAULT_CONFIG)
