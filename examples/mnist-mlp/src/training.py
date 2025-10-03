from typing import Iterable 
from torch import Tensor
from torchsystem.depends import Depends, Provider
from torchsystem.services import Service, Producer, event 
from mltracker.ports import Models
from src.classifier import Classifier

provider = Provider()
producer = Producer() 
service = Service(provider=provider)

def device() -> str:...
def models() -> Models:...

@event
class Trained:
    model: Classifier 
    results: dict[str, Tensor] 

@event
class Evaluated:
    model: Classifier
    results: dict[str, Tensor]

@service.handler
def train(model: Classifier, loader: Iterable[tuple[Tensor, Tensor]], device: str = Depends(device)):
    model.phase = 'train'
    for inputs, targets in loader: 
        inputs, targets = inputs.to(device), targets.to(device)  
        loss = model.fit(inputs, targets)
    producer.dispatch(Trained(model, {"loss": loss}))

@service.handler
def evaluate(model: Classifier, loader: Iterable[tuple[Tensor, Tensor]], device: str = Depends(device)):
    model.phase = 'evaluation'
    for inputs, targets in loader: 
        inputs, targets = inputs.to(device), targets.to(device)  
        loss = model.evaluate(inputs, targets)
    producer.dispatch(Evaluated(model, {"loss": loss})) 