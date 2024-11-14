from typing import Sequence
from datetime import datetime
from dataclasses import dataclass
from pybondi import Event
from torchsystem.aggregate import Aggregate, Loader

@dataclass
class Trained(Event):
    '''
    The Trained event is used to signal that the aggregate has been trained.
    over a sequence of loaders.
    '''
    epoch: int
    start: datetime
    end: datetime
    loaders: Sequence[Loader]
    aggregate: Aggregate
    
@dataclass
class Evaluated(Event):
    '''
    The Evaluated event is used to signal that the aggregate has been evaluated
    over a sequence of loaders.
    '''
    epoch: int
    start: datetime
    end: datetime
    loaders: Sequence[Loader]
    aggregate: Aggregate

@dataclass
class Iterated(Event):
    '''
    The Iterated event is used to signal that the aggregate has been iterated
    over a sequence of loaders.
    '''
    epoch: int
    start: datetime
    end: datetime
    loaders: Sequence[tuple[str, Loader]]
    aggregate: Aggregate