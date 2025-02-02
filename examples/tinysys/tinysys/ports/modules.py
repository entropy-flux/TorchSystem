from abc import ABC, abstractmethod
from typing import Any
from typing import Optional
from attrs import define

@define
class Module:
    type: str
    hash: str
    name: str
    epoch: int
    arguments: dict[str, Any]

class Modules(ABC):
    
    @abstractmethod
    def list(self, type: str) -> list[Module]:...

    @abstractmethod
    def last(self, type: str) -> Optional[Module]:...
    
    @abstractmethod
    def put(self, module: Module):...

    @abstractmethod
    def clear(self):...