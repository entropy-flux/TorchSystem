from typing import Sequence

from torch import cuda
from torch import inference_mode
from torchsystem import Depends 
from torchsystem.depends import Provider
from torchsystem.services import Service
from torchsystem.services import Producer, event

from tinysys.domain import Tensor
from tinysys.domain import Model
from tinysys.domain import Loader
from tinysys.domain import Metrics
  
provider = Provider()
service = Service(provider=provider)
producer = Producer()

def device() -> str:...

@service.handler
def iterate(model: Model, loaders: Sequence[tuple[str, Loader]], metrics: Metrics):
    for phase, loader in loaders:
        train(model, loader, metrics) if phase == 'train' else validate(model, loader, metrics)
        metrics.reset()
    model.epoch += 1
    producer.dispatch(Iterated(model, loaders))

@service.handler
def train(model: Model, loader: Loader, metrics: Metrics, device: str = Depends(device)):
    model.phase = 'train'
    for batch, (inputs, targets) in enumerate(loader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        predictions, loss = model.fit(inputs, targets)
        metrics.update(batch, loss, predictions, targets)
    results = metrics.compute()
    producer.dispatch(Trained(model, results))

@service.handler
def validate(model: Model, loader: Loader, metrics: Metrics, device: str = Depends(device)):
    with inference_mode():
        model.phase = 'evaluation'
        for batch, (inputs, targets) in enumerate(loader, start=1):
            inputs, targets = inputs.to(device), targets.to(device)
            predictions, loss = model.evaluate(inputs, targets)
            metrics.update(batch, loss, predictions, targets)
        results = metrics.compute()
        producer.dispatch(Validated(model, results))

@event
class Iterated:
    model: Model
    loaders: Sequence[tuple[str, Loader]]

@event
class Trained:
    model: Model
    results: dict[str, Tensor]

@event
class Validated:
    model: Model
    results: dict[str, Tensor]