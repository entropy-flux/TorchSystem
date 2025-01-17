from typing import Any
from typing import Callable

class Metric:
    name: str
    value: Any

    def __call__(self, **kwargs):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

class Metrics[T: Metric]:
    def __init__(self, *handlers: T):
        self.handlers = {handler.name: handler for handler in handlers}

    def __call__(self, **kwargs) -> tuple[Metric]:
        return tuple(handler(**kwargs) for handler in self.handlers.values())
    
    def __iter__(self):
        return iter(self.handlers.values())

    def handler(self, handler: T) -> T:
        self.handlers[handler.name] = handler
        return handler
    
    def __getitem__(self, key: int) -> T:
        return self.handlers[key].value
    
class Callback:
    def __init__(self, *metrics: Metric):
        self.metrics = Metrics(*metrics)
        self.handlers = []

    def __call__(self, *args, **kwargs):
        [handler(*args, **kwargs) for handler in self.handlers]

    def handler(self, function: Callable) -> Callable:
        self.handlers.append(function)

    def __iter__(self):
        return iter(self.metrics.handlers.values())
    
    def reset(self):
        for handler in self.metrics:
            handler.reset()