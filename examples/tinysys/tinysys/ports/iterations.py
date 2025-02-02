from abc import ABC, abstractmethod
from typing import Any
from attrs import define

@define
class Iteration:
    hash: str
    phase: str
    epoch: int
    arguments: dict[str, Any]

class Iterations(ABC):

    @abstractmethod
    def add(self, iteration: Iteration):...
    
    @abstractmethod
    def put(self, iteration: Iteration):...

    @abstractmethod
    def list(self) -> list[Iteration]:...
    
    @abstractmethod
    def clear(self):...