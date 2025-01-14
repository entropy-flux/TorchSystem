from typing import Literal
from typing import Callable
from torch import Tensor
from torch import inference_mode
from torch.nn import Module
from torch.optim import Optimizer
from torchsystem import Aggregate
from torchsystem import Loader
from torchsystem.settings import BaseSettings
from torchsystem.schemas import Metric

# Sometimes more complicated settings are needed.
class ClassifierSettings(BaseSettings): 
    device: str = 'cpu' # Since this is a pydantic-settings class, you can define this values in a .env file if you prefer.

class Classifier(Aggregate):
    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer, settings: ClassifierSettings = None):
        super().__init__()
        self.epoch = 0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = settings.device
    
    @property
    def id(self) -> None:
        return self.model.__class__.__name__

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)
    
    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        return self.criterion(output, target)

    def fit(self, loader: Loader, callback: Callable):
        self.phase = 'train'
        for batch, (input, target) in enumerate(loader, start=1):
            input, target = input.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(input)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            for name, value in callback(loss=loss, output=output, target=target):
                self.publish(Metric(name, value, self.phase, self.epoch, batch), topic='metrics')
                
            # You define the callback you want outside the aggregate. This returns
            # a Sequence[tuple[str, float]] with the name of the metric and its value.

    @inference_mode()
    def evaluate(self, loader: Loader, callback: Callable):
        self.phase = 'evaluation'
        for batch, (input, target) in enumerate(loader, start=1):
            input, target = input.to(self.device), target.to(self.device)
            output = self(input)
            loss = self.loss(output, target)
            for name, value in callback(loss=loss, output=output, target=target):
                self.publish(Metric(name, value, self.phase, self.epoch, batch), topic='metrics')