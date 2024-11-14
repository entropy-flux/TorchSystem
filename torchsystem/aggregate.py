from abc import ABC, abstractmethod
from typing import Any
from typing import Iterator
from typing import Protocol
from typing import overload
from typing import Callable
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset
from pybondi.aggregate import Root

class Loader(Protocol):
    '''
    Interface for the DataLoader class.     
    '''
    dataset: Dataset
    
    def __iter__(self) -> Iterator[Any]:...

    @overload
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:...


class Aggregate(Module, ABC):

    def __init__(self, id: Any):
        super().__init__()
        self.root = Root(id=id)
        self.epoch = 0
    
    @property
    def phase(self) -> str:
        return 'train' if self.training else 'evaluation'
    
    @phase.setter
    def phase(self, value: str):
        self.train() if value == 'train' else self.eval()

    @abstractmethod
    def fit(self, data: Loader, callback: Callable): ...

    @abstractmethod
    def evaluate(self, data: Loader, callback: Callable): ...

    def iterate(self, data: Loader, callback: Callable):
        self.fit(data, callback) if self.training else self.evaluate(data, callback)