from typing import Iterable 
from torch import Tensor
from torch import inference_mode
from torchsystem.depends import Depends, Provider
from torchsystem.services import Service, Producer, event 
from mltracker.ports import Models
from src.metrics import Metrics
from src.classifier import Classifier

provider = Provider()
producer = Producer() 
service = Service(provider=provider)

def device() -> str:...
def models() -> Models:...
def location() -> str:...
  
@service.handler
def train(model: Classifier, loader: Iterable[tuple[Tensor, Tensor]], device: str = Depends(device)):
    model.phase = 'train'
    for batch, (inputs, targets) in enumerate(loader, start=1): 
        inputs, targets = inputs.to(device), targets.to(device)  
        predictions, loss = model.fit(inputs, targets)
        model.metrics.update(batch, loss, predictions, targets)
    results = model.metrics.compute()
    producer.dispatch(Trained(model, results, loader))


@service.handler
def evaluate(model: Classifier, loader: Iterable[tuple[Tensor, Tensor]], device: str = Depends(device)):
    model.phase = 'evaluation'
    with inference_mode():
        for batch, (inputs, targets) in enumerate(loader, start=1): 
            inputs, targets = inputs.to(device), targets.to(device)  
            predictions, loss = model.evaluate(inputs, targets)
            model.metrics.update(batch, loss, predictions, targets)
        results = model.metrics.compute()
        producer.dispatch(Evaluated(model, results, loader))

@event
class Trained:
    model: Classifier 
    results: dict[str, Tensor] 
    loader: Iterable[tuple[Tensor, Tensor]]

@event
class Evaluated:
    model: Classifier
    results: dict[str, Tensor]
    loader: Iterable[tuple[Tensor, Tensor]]
