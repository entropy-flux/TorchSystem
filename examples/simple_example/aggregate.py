from typing import Callable
from typing import Sequence
from typing import Any
from torch import Tensor
from torch import inference_mode
from torch.nn import Module
from torch.optim import Optimizer
from torchsystem import Aggregate
from torchsystem.loaders import Loader
from torchsystem.settings import BaseSettings, Settings
from torchsystem.schemas import Metric

class ClassifierSettigs(BaseSettings):
    device: str

class Classifier(Aggregate):
    def __init__(self, id: str, model: Module, criterion: Module, optimizer: Optimizer, settings: Settings[ClassifierSettigs]):
        super().__init__(id)
        self.epochs = 0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.settings = settings

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)
    
    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        return self.criterion(output, target)

    def fit(self, loader: Loader, callback: Callable[..., Sequence[Metric]]):
        for batch, (input, target) in enumerate(loader, start=1):
            input, target = input.to(self.settings.aggregate.device), target.to(self.settings.aggregate.device)
            self.optimizer.zero_grad()
            output = self.forward(input)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            for metric in callback(loss=loss.item(), output=output, target=target):
                self.publish(metric.name, Metric(metric.name, metric.value, batch, self.epochs, self.phase))

    @inference_mode()
    def evaluate(self, loader: Loader, callback: Callable[..., Sequence[Metric]]):
        for batch, (input, target) in enumerate(loader, start=1):
            input, target = input.to(self.settings.aggregate.device), target.to(self.settings.aggregate.device)
            output = self.forward(input)
            loss = self.loss(output, target)
            callback(loss=loss.item(), output=output, target=target)
            for metric in callback(loss=loss.item(), output=output, target=target):
                self.publish(metric.name, Metric(metric.name, metric.value, batch, self.epochs, self.phase))

        