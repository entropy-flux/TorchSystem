from typing import Callable
from torch import compile
from torchsystem import Service
from torchsystem.aggregate import Aggregate
from torchsystem.settings import Settings

class Compiler[T: Aggregate]:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.services = Service()

    def factory(self, wrapped: Callable[..., T]) -> T:
        def wrapper(*args, **kwargs) -> T:
            aggregate = wrapped(*args, **kwargs, settings=self.settings)
            aggregate.events = self.services.events
            aggregate.publisher = self.services.publisher
        self.builder = wrapper
        return wrapper
    
    def compile(self, *args, **kwargs) -> T:
        return self.builder(*args, **kwargs)