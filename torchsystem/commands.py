from typing import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pybondi import Command
from pybondi.callbacks import Callback
from torchsystem.aggregate import Aggregate
from torchsystem.aggregate import Loader
from torchsystem.events import Trained, Evaluated, Iterated

@dataclass
class Train(Command):
    '''
    The Train command is used to train the aggregate using the sequence of loaders provided. 
    It bumps the epoch after training the aggregate.
    '''
    aggregate: Aggregate
    loaders: Sequence[Loader]
    callback: Callback
    
    def execute(self):
        self.aggregate.phase = 'train'
        self.callback.set('phase', self.aggregate.phase)
        self.callback.set('epoch', self.aggregate.epoch)
        start = datetime.now(timezone.utc)
        for loader in self.loaders:
            self.aggregate.fit(loader, self.callback)
        end = datetime.now(timezone.utc)
        self.aggregate.root.publish(event=Trained(
            epoch=self.aggregate.epoch,
            start=start,
            end=end,
            loaders=self.loaders,
            aggregate=self.aggregate
        ))
        self.aggregate.epoch += 1

@dataclass
class Evaluate(Command):
    '''
    The Evaluate command is used to evaluate the aggregate using the sequence of loaders provided. 
    The aggregate is put on evaluation mode. It does not bump the epoch since it is not training the aggregate. 
    '''
    aggregate: Aggregate
    loaders: Sequence[Loader]
    callback: Callback
    
    def execute(self):
        self.aggregate.phase = 'evaluation'
        self.callback.set('phase', self.aggregate.phase)
        self.callback.set('epoch', self.aggregate.epoch)
        start = datetime.now(timezone.utc)
        for loader in self.loaders:
            self.aggregate.evaluate(loader, self.callback)
        end = datetime.now(timezone.utc)
        self.aggregate.root.publish(event=Evaluated(
            epoch=self.aggregate.epoch,
            start=start,
            end=end,
            loaders=self.loaders,
            aggregate=self.aggregate
        ))

@dataclass
class Iterate(Command):
    '''
    The Iterate command is used to iterate over the sequence of loaders provided.
    It determines the phase of the aggregate and calls the fit or evaluate method accordingly.
    After iterating over the loaders, it bumps the epoch.
    '''
    aggregate: Aggregate
    loaders: Sequence[tuple[str, Loader]]
    callback: Callback
    
    def execute(self):
        self.callback.set('epoch', self.aggregate.epoch)
        start = datetime.now(timezone.utc)
        for phase, loader in self.loaders:
            self.aggregate.phase = phase
            self.callback.set('phase', self.aggregate.phase )
            self.aggregate.iterate(loader, self.callback)
        end = datetime.now(timezone.utc)
        self.aggregate.root.publish(event=Iterated(
            epoch=self.aggregate.epoch,
            start=start,
            end=end,
            loaders=self.loaders,
            aggregate=self.aggregate
        ))
        self.aggregate.epoch += 1