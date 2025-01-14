from typing import Any
from typing import Optional
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Message[T]:
    payload: T
    sender: Optional[Any] = None
    headers: Optional[dict] = None
    timestamp: Optional[datetime] = None

@dataclass
class Metric:
    name: str
    value: Any
    phase: Optional[str] = None
    epoch: Optional[int] = None
    batch: Optional[int] = None