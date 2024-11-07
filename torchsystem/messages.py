from typing import Any
from dataclasses import dataclass

@dataclass
class Metric:
    name: str
    value: Any
    batch: int
    epoch: int
    phase: str