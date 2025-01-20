from torch import Tensor
from typing import Protocol
from typing import Iterator
from typing import Any
from typing import overload

class Loader(Protocol):
    """
    The Loader protocol defines the interface for loading data. DataLoaders are an infrastructure
    concern and lack of PEP 484 support for them is a known issue. The Loader protocol provides
    a simple interface to be used in the service layer without mixing infrastructure concerns with
    the business logic.

    Example:

        .. code-block:: python

        from torchsystem import Loader

        def iterate(loader: Loader):
            for input, target in loader:
                input, target = input.to(device), target.to(device) # input and target have type hints.
    """

    def __iter__(self) -> Iterator[Any]:...

    @overload
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:...

    @overload
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:...