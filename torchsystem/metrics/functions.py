from torch import Tensor
from torch import argmax
    
def accuracy(predictions: Tensor, target: Tensor) -> float:
    return (predictions == target).float().mean().item()

def predictions(output: Tensor) -> Tensor:
    return argmax(output, dim=1)