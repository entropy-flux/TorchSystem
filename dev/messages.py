from uuid import UUID
from typing import Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from torchsystem.aggregate import Loader, Aggregate
from torchsystem.loaders import Loaders
from mlregistry import get_metadata, get_date_hash

from typing import Protocol
class Aggregate(Protocol):
    epoch: int
    model: Any
    criterion: Any
    optimizer: Any

@dataclass
class Message:
    def dump(self):
        return asdict(self)

@dataclass
class Metric(Message):
    name: str
    value: Any
    batch: int
    epoch: int
    phase: str

@dataclass
class Experiment(Message):
    id: Any
    name: str

@dataclass
class Model(Message):
    id: Any
    hash: str
    name: str
    epochs: int
    parameters: dict[str, Any]    

    @staticmethod
    def create(aggregate: Aggregate):
        metadata = get_metadata(aggregate.model)
        return Model(
            id=None,
            hash=metadata.hash,
            name=metadata.name,
            epochs=aggregate.epoch,
            parameters=metadata.arguments
        )

@dataclass
class Criterion(Message):
    hash: str
    name: str
    parameters: dict[str, Any]

    @staticmethod
    def create(aggregate: Aggregate):
        metadata = get_metadata(aggregate.criterion)
        return Criterion(
            hash=metadata.hash,
            name=metadata.name,
            parameters=metadata.arguments
        )
    

@dataclass
class Optimizer(Message):
    hash: str
    name: str
    parameters: dict[str, Any]

    @staticmethod
    def create(aggregate: Aggregate):
        metadata = get_metadata(aggregate.optimizer)
        return Optimizer(
            hash=metadata.hash,
            name=metadata.name,
            parameters=metadata.arguments
        )

@dataclass
class Dataset(Message):
    hash: str
    name: str
    parameters: dict[str, Any]

    @staticmethod
    def create(loader: Loader):
        metadata = get_metadata(loader.dataset)
        return Dataset(
            hash=metadata.hash,
            name=metadata.name,
            parameters=metadata.arguments
        )

@dataclass
class Iteration(Message):
    phase: str
    dataset: Dataset
    parameters: dict[str, Any]
    
    @staticmethod
    def create(phase: str, loader: Loader):
        metadata = get_metadata(loader)
        return Iteration(
            phase=phase,
            dataset=Dataset.create(loader),
            parameters=metadata.arguments
        )
    
@dataclass
class Transaction(Message):
    epochs: tuple[int, int]
    hash: str
    start: datetime
    end: datetime
    criterion: Criterion
    optimizer: Optimizer
    iterations: list[Iteration]

    @staticmethod
    def create(aggregate: Aggregate, loader: Loaders):
        start = datetime.now(timezone.utc)
        return Transaction(
            epochs=(aggregate.epoch, aggregate.epoch),
            hash=get_date_hash(start),
            start=start,
            end=None,
            criterion=Criterion.create(aggregate),
            optimizer=Optimizer.create(aggregate),
            iterations=[Iteration.create(phase, loader) for phase, loader in loader]
        )