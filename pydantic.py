class BaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def validator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def root_validator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def Field(default=None, **kwargs):
    return default

ConfigDict = dict
