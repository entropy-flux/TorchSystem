from abc import ABC, abstractmethod
from attrs import define
from tinysys.ports.models import Models

@define
class Experiment:
    id: str
    name: str
    models: Models

class Experiments(ABC):

    @abstractmethod
    def create(self, name: str) -> Experiment:...

    @abstractmethod
    def read(self, id: str) -> Experiment:...

    @abstractmethod
    def update(self, id: str, name: str) -> Experiment:...

    @abstractmethod
    def delete(self, id: str) -> None:...

    @abstractmethod
    def list(self) -> list[Experiment]:...