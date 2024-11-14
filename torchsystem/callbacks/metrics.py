from dataclasses import dataclass
from typing import Any
from typing import Any
from torch import Tensor
from torch import argmax

@dataclass
class Metric:
    name: str
    value: Any
    batch: int
    epoch: int
    phase: str
    
def accuracy(predictions: Tensor, target: Tensor) -> float:
    return (predictions == target).float().mean().item()

def predictions(output: Tensor) -> Tensor:
    return argmax(output, dim=1)