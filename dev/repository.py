from pybondi import Repository as Base
from pybondi.aggregate import Aggregate
from torchsystem.storage import Models, Criterions, Optimizers, Datasets
from typing import Any, Protocol

class Root(Protocol):
    id: str

class Aggregate(Protocol):
    root: Root
    epoch: int
    model: Any
    criterion: Any
    optimizer: Any

class Repository(Base):
    def __init__(self, folder: str):
        super().__init__()
        self.models = Models(folder)
        self.criterions = Criterions(folder)
        self.optimizers = Optimizers(folder)
        self.datasets = Datasets()
        self.epochs = dict[str, int]()

    def store(self, aggregate: Aggregate):
        self.epochs[aggregate.root.id] = aggregate.epoch
        self.models.save(aggregate.model)
        self.criterions.save(aggregate.criterion)
        self.optimizers.save(aggregate.optimizer)

    def restore(self, aggregate: Aggregate):
        aggregate.epoch = self.epochs[aggregate.root.id]
        self.models.weights.restore(aggregate.model)
        self.criterions.weights.restore(aggregate.criterion)
        self.optimizers.weights.restore(aggregate.optimizer)