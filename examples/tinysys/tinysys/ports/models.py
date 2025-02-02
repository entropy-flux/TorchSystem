from abc import ABC, abstractmethod
from typing import Optional
from attrs import define 

from tinysys.ports.metrics import Metrics
from tinysys.ports.modules import Modules
from tinysys.ports.iterations import Iterations

@define
class Model:
    id: str
    name: str
    epoch: int 
    metrics: Metrics
    modules: Modules
    iterations: Iterations

class Models(ABC):

    @abstractmethod
    def create(self, id: str, name: str) -> Model:
        ...

    @abstractmethod
    def read(self, id: str) -> Optional[Model]:
        ...
        
    @abstractmethod
    def update(self, id: str, epoch: int) -> Model:
        ...

    @abstractmethod
    def delete(self, id: str):
        ...

    @abstractmethod
    def clear(self):
        ...

    @abstractmethod
    def list(self) -> list[Model]:
        ...