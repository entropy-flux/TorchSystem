from typing import Any
from typing import Iterator
from typing import overload
from typing import Protocol
from typing import runtime_checkable
from torch import Tensor

@runtime_checkable
class Loader(Protocol):

    def __iter__(self) -> Any:...

    @overload
    def __iter__(self) -> Iterator[Tensor]:...

    @overload
    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:...