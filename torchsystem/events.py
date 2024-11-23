from typing import Sequence
from datetime import datetime
from dataclasses import dataclass
from pybondi import Event
from pybondi.events import Event, Added, RolledBack, Commited
from torchsystem.aggregate import Aggregate, Loader

@dataclass
class Trained[T: Aggregate](Event):
    '''
    The Trained event is used to signal that the aggregate has been trained.
    over a sequence of loaders.
    '''
    epoch: int
    start: datetime
    end: datetime
    loaders: Sequence[Loader]
    aggregate: T
    
@dataclass
class Evaluated[T: Aggregate](Event):
    '''
    The Evaluated event is used to signal that the aggregate has been evaluated
    over a sequence of loaders.
    '''
    epoch: int
    start: datetime
    end: datetime
    loaders: Sequence[Loader]
    aggregate: T

@dataclass
class Iterated[T: Aggregate](Event):
    '''
    The Iterated event is used to signal that the aggregate has been iterated
    over a sequence of loaders.
    '''
    epoch: int
    start: datetime
    end: datetime
    loaders: Sequence[tuple[str, Loader]]
    aggregate: T