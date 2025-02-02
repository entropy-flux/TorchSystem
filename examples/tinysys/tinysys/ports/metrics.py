from abc import ABC, abstractmethod
from attrs import define

@define
class Metric:
    name: str
    value: float
    phase: str
    epoch: int

class Metrics(ABC):

    @abstractmethod
    def add(self, metric: Metric):...
    
    @abstractmethod
    def list(self) -> list[Metric]:...

    @abstractmethod
    def clear(self):...