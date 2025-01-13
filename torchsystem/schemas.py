from typing import Any
from typing import Optional
from dataclasses import dataclass

@dataclass
class Message[T]:
    sender: Any
    topic: str
    payload: T

@dataclass
class Metric:
    name: str
    value: Any
    batch: Optional[int] = None
    epoch: Optional[int] = None
    phase: Optional[str] = None