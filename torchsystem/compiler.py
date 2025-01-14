from typing import Callable
from typing import Any
from torch import compile
from torchsystem.settings import Settings
from torchsystem.aggregate import Aggregate

class Compiler[T: Aggregate]:
    def __init__(self, settings: Settings):
        self.pipeline = list[Callable[..., Any]]()
        self.settings = settings

    def compile(self, compilable: T) -> T:
        return compile(
            compilable,
            fullgraph=self.settings.compiler.fullgraph,
            dynamic=self.settings.compiler.dynamic,
            backend=self.settings.compiler.backend,
            mode=self.settings.compiler.mode,
            disable=self.settings.compiler.disable
        )

    def step(self, callable: Callable[..., Any]) -> Callable[..., Any]:
        self.pipeline.append(callable)
        return callable
    
    def __call__(self, *args, **kwargs) -> T:
        result = None
        for step in self.pipeline:
            result = step(*args, **kwargs) if not result else step(result)
        return result