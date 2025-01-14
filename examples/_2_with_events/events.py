from dataclasses import dataclass
from torchsystem import Event
from torchsystem.aggregate import Aggregate

@dataclass
class Stored[T: Aggregate]:
    aggregate: T

@dataclass
class Restored[T: Aggregate]:
    aggregate: T

