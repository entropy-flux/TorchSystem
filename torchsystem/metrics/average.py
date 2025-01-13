from typing import Any
from typing import Literal
from typing import Optional
from torch import Tensor
from torchsystem.schemas import Metric
from torchsystem.metrics.functions import accuracy, predictions

class Cumulative:
    """
    A class to calculate the cumulative average of a value or sum of values.

    Attributes:
        samples (int): The number of samples added to the average.
        value (Any): The current average
    """

    def __init__(self):
        self.samples = 0
        self.value = 0

    def reset(self):
        """
        Reset the average to zero.
        """
        self.samples = 0
        self.value = 0

    def update(self, value: Any, samples: int = 1):
        """
        Update the average with a new value or sum of values and number of samples.

        Args:
            value (Any): The new value (sum of values) to add to the average.
            samples (Optional[int]): The number of samples to add to the average. Defaults to 1.
        """
        self.value = (self.value * self.samples + value) / (self.samples + samples)
        self.samples += samples

class Loss:
    def __init__(self):
        self.average = Cumulative()

    def __call__(self, *, loss: float, **kwargs) -> Metric:
        self.average.update(loss)
        return Metric(name='loss', value=self.average.value)

    def reset(self):
        self.average.reset()

class Accuracy:
    def __init__(self):
        self.average = Cumulative()

    def __call__(self, *, output: Tensor, target: Tensor, **kwargs):
        self.average.update(accuracy(predictions(output), target))
        return Metric(name='accuracy', value=self.average.value)

    def reset(self):
        self.average.reset()