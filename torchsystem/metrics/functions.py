from torch import Tensor
from torch import argmax
    
def accuracy(predictions: Tensor, target: Tensor) -> float:
    """
    Calculate the accuracy of the predictions with respect to the target.

    Args:
        predictions (Tensor): The predictions made by the model.
        target (Tensor): The target values.

    Returns:
        float: The accuracy of the predictions.
    """
    return (predictions == target).float().mean().item()

def predictions(output: Tensor) -> Tensor:
    """
    Get the predictions from the output of the model. The predictions are the class with the highest probability.

    Args:
        output (Tensor): The raw output of the model.

    Returns:
        Tensor: The predictions.
    """
    return argmax(output, dim=1)