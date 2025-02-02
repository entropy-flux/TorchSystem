from torchsystem import Depends 
from torchsystem.depends import Provider
from torchsystem.services import Consumer  
from torchsystem.registry import gethash, getname, getarguments

from tinysys.domain import Repository
from tinysys.ports import structure
from tinysys.ports.models import Models
from tinysys.ports.modules import Module
from tinysys.ports.metrics import Metric
from tinysys.ports.iterations import Iteration
from tinysys.services.training import (
    Trained,
    Validated,
    Iterated
)

provider = Provider()
consumer = Consumer(provider=provider) 

def models() -> Models:...

def repository() -> Repository:...

@consumer.handler
def handle_metrics(event: Trained | Validated, models: Models = Depends(models)):
    model = models.read(id=event.model.id)
    for name, metric in event.results.items():
        model.metrics.add(structure({
            'name': name,
            'value': metric.item(),
            'phase': event.model.phase,
            'epoch': event.model.epoch
        }, Metric))

@consumer.handler
def handle_nn(event: Iterated, models: Models = Depends(models)):
    model = models.read(id=event.model.id)
    model.modules.put(structure({
        'type': 'nn',
        'hash': gethash(event.model.nn),
        'name': getname(event.model.nn),
        'epoch': event.model.epoch, 
        'arguments': getarguments(event.model.nn)
    }, Module))

@consumer.handler
def handle_criterion(event: Iterated, models: Models = Depends(models)):
    model = models.read(id=event.model.id)
    model.modules.put(structure({
        'type': 'criterion',
        'hash': gethash(event.model.criterion),
        'name': getname(event.model.criterion),
        'epoch': event.model.epoch, 
        'arguments': getarguments(event.model.criterion)
    }, Module))

@consumer.handler
def handle_optimizer(event: Iterated, models: Models = Depends(models)):
    model = models.read(id=event.model.id)
    model.modules.put(structure({
        'type': 'optimizer',
        'hash': gethash(event.model.optimizer),
        'name': getname(event.model.optimizer),
        'epoch': event.model.epoch, 
        'arguments': getarguments(event.model.optimizer)
    }, Module))

@consumer.handler
def handle_iterations(event: Iterated, models: Models = Depends(models)):
    model = models.read(id=event.model.id)
    for phase, loader in event.loaders:
        model.iterations.put(structure({
            'hash': gethash(loader),
            'epoch': event.model.epoch,
            'phase': phase,
            'arguments': getarguments(loader)
        }, Iteration))

@consumer.handler
def handle_epoch(event: Iterated, models: Models = Depends(models)):
    models.update(id=event.model.id, epoch=event.model.epoch)

@consumer.handler
def save_weights(event: Iterated, repository: Repository = Depends(repository)):
    repository.store(event.model)