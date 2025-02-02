from typing import Protocol
from typing import Iterator
from typing import Sequence

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer 
from torchsystem.domain import Events

class Model(Protocol):
    id: str
    epoch: int
    phase: str
    nn: Module
    criterion: Module
    optimizer: Optimizer
    events: Events

    def fit(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        ...

    def evaluate(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        ...


class Loader(Protocol):

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        ...

class Metrics(Protocol):

    def update(self, *args: Tensor) -> None:
        ...

    def compute(self) -> dict[str, Tensor]:
        ...

    def reset(self) -> None:
        ...

class Repository(Protocol):

    def store(self, model: Model) -> None:
        ...

    def restore(self, model: Model):
        ...