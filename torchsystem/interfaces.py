from typing import Any
from typing import Iterator
from typing import overload
from typing import Protocol
from typing import runtime_checkable
from torch import Tensor

@runtime_checkable
class Loader(Protocol):
    """
    A runtime checkable protocol for pytorch data loaders with type hints. This fixes the issue with
    the `torch.utils.data.DataLoader` class not having type hints for the `__iter__` method. Use this
    as an interface for passing around data loaders with type hints.

    Example:

        .. code-block:: python

        from torchsystem.loaders import Loader

        def train(model: Module, loader: Loader):
            for input, target in loader:
                input, target = input.to(device), target.to(device)  # Now these are type hinted
                ...
    """

    def __iter__(self) -> Any:...

    @overload
    def __iter__(self) -> Iterator[Tensor]:...

    @overload
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:...


@runtime_checkable
class Metrics(Protocol):
    
    def __call__(self, **kwargs) -> Any:...

    def reset(self):...