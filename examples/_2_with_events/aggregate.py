from uuid import UUID
from typing import Callable
from torch import Tensor
from torch import inference_mode
from torch.nn import Module
from torch.optim import Optimizer
from torchsystem import Aggregate
from torchsystem import Loader
from torchsystem import Settings
from torchsystem.storage import get_hash
from torchsystem.settings import AggregateSettings

class ClassifierSettings(AggregateSettings): 
    device: str

class Classifier(Aggregate):
    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer, settings: Settings[ClassifierSettings] = None):
        super().__init__()
        self.epoch = 0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = settings.aggregate.device
    
    @property
    def id(self) -> str:
        return get_hash(self.model)

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
            metrics = callback(loss=loss.item(), output=output, target=target)
            if batch % 200 == 0:
                self.publish((batch, [(name, value) for name, value in metrics]), topic='metrics')

    @inference_mode()
    def evaluate(self, loader: Loader, callback: Callable):
        self.phase = 'evaluation'
        for batch, (input, target) in enumerate(loader, start=1):
            input, target = input.to(self.device), target.to(self.device)
            output = self(input)
            loss = self.loss(output, target)
            metrics = callback(loss=loss.item(), output=output, target=target)
            if batch % 200 == 0:
                self.publish((batch, [(name, value) for name, value in metrics]), topic='metrics')