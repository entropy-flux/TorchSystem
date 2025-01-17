from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor
from torchsystem.metrics.callback import Metric
from torchsystem.metrics.functions import accuracy, predictions

### TODO: This has to be refactored without altering the usage
### of the Callback class.

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

class Average(ABC):
    """
    An abstract base class to calculate the average of a metric. It should be inherited by
    the user to create a custom average metric.

    Example:

        .. code-block:: python

        from torchsystem.metrics.average import Average

        class Loss(Average):
            name = 'loss'
            def __init__(self):
                self.average = Cumulative()

            def __call__(self, *, loss: float, **kwargs) -> Metric:
                self.average.update(loss)
                return self

            def reset(self):
                self.average.reset()
    """
    name: str
    average: Cumulative

    @property
    def value(self) -> Any:
        return self.average.value

    @abstractmethod
    def __call__(self, **kwargs) -> tuple[str, Any]:...

    @abstractmethod
    def reset(self):...


class Loss(Average):
    """
    A class to calculate the average loss. Designed to be used with the `Callback` class.

    Attributes:
        average (Cumulative): The cumulative average of the loss.

    Methods:
        __call__:
            Update the average loss and return the name and value of the metric.

    Example:

        .. code-block:: python

        from torchsystem.metrics import Callback
        from torchsystem.metrics.average import Loss, Accuracy

        callback = Callback(Loss(), Accuracy())
        name, value = next(callback(callback(predictions=predictions, loss=0.5)))
        print(name, value) # loss 0.5
    """
    name = 'loss'
    def __init__(self):
        self.average = Cumulative()

    def __call__(self, *, loss: float, **kwargs) -> Metric:
        """
        Update the average loss and return the name and value of the metric

        Args:
            loss (float): The loss value to add to the average.

        Returns:
            Metric: The name and value of the updated metric.
        """
        self.average.update(loss)
        return self

    def reset(self):
        self.average.reset()
        

class Accuracy(Average):
    """
    A class to calculate the average accuracy. The accuracy is calculated as the number of correct
    predictions divided by the total number of predictions.
    
    Attributes:
        average (Cumulative): The cumulative average of the accuracy.

    Methods:
        __call__:
            Update the average accuracy and return the name and value of the metric.

    Example:
    
        .. code-block:: python

        from torchsystem.metrics import Callback
        from torchsystem.metrics.average import Loss, Accuracy

        callback = Callback(Accuracy(), Loss())
        metric = next(callback(predictions=predictions, target=target, loss=0.5))   
        print(metric.name, metric.value) # accuracy 0.5
    """
    name = 'accuracy'

    def __init__(self):
        self.average = Cumulative()

    def __call__(self, *, predictions: Tensor, target: Tensor, **kwargs) -> Metric:
        """
        Update the average accuracy and return the name and value of the metric.

        Args:
            predictions (Tensor):  The predicted values.
            target (Tensor): The target values.

        Returns:
            Metric: 
        """
        self.average.update(accuracy(predictions, target))
        return self

    def reset(self):
        self.average.reset()