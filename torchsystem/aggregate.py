from abc import ABC, abstractmethod
from typing import Any
from typing import Iterator
from typing import Protocol
from typing import overload
from typing import Literal
from typing import Callable
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset
from pybondi.aggregate import Root

class Loader(Protocol):
    '''
    Interface for the DataLoader class.     
    '''
    dataset: Dataset
    
    def __iter__(self) -> Iterator[Any]:...

    @overload
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:...


class Aggregate(Module, ABC):
    """
    A base class for aggregate modules, managing model training and evaluation phases,
    root identification, and epoch tracking.

    Attributes:
        root (Root): The aggregate root, initialized with an identifier.
        epoch (int): The current epoch count.
    """

    def __init__(self, id: Any):
        super().__init__()
        self.root = Root(id=id)
        self.epoch = 0

    @property
    def id(self) -> Any:
        """
        Returns the unique identifier of the aggregate root.

        Returns:
            Any: The ID of the root.
        """
        return self.root.id
    
    @property
    def phase(self) -> Literal['train', 'evaluation']:
        """
        Returns the current phase of the model: 'train' or 'evaluation'.

        Returns:
            str: The current phase ('train' or 'evaluation').
        """
        return 'train' if self.training else 'evaluation'
    
    @phase.setter
    def phase(self, value: Literal['train', 'evaluation']):
        """
        Sets the phase of the model to either 'train' or 'evaluation'.

        Args:
            value (str): The phase to set ('train' or 'evaluation').
        """
        self.train() if value == 'train' else self.eval()

    def fit(self, data: Loader, callback: Callable):
        """
        Abstract method for fitting the model to the given data. Implement
        this method in a subclass to define your training logic.

        Args:
            data (Loader): The data loader providing the training data.
            callback (Callable): A callback function to be executed during training.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def evaluate(self, data: Loader, callback: Callable):
        """
        Abstract method for evaluating the model on the given data. Implement
        this method in a subclass to define your evaluation logic.

        Args:
            data (Loader): The data loader providing the evaluation data.
            callback (Callable): A callback function to be executed during evaluation.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def iterate(self, data: Loader, callback: Callable):
        """
        Executes the appropriate method based on the current phase. It iterates
        the model over the given data and calling the provided callback function.

        - Calls `fit` if the model is in the training phase.
        - Calls `evaluate` if the model is in the evaluation phase.

        Args:
            data (Loader): The data loader providing the data.
            callback (Callable): A callback function to be executed during the process.
        """
        self.fit(data, callback) if self.training else self.evaluate(data, callback)