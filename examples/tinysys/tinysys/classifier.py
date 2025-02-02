from torch import Tensor
from torch import argmax
from torch.nn import Module
from torch.nn import Flatten
from torch.optim import Optimizer
from torchsystem import Aggregate
from torchsystem.registry import gethash

class Classifier(Aggregate):
    def __init__(self, nn: Module, criterion: Module, optimizer: Optimizer):
        super().__init__()
        self.nn = nn
        self.epoch = 0
        self.criterion = criterion
        self.optimizer = optimizer
        self.flatten = Flatten()

    @property
    def id(self) -> str:
        return gethash(self.nn)

    def forward(self, input: Tensor) -> Tensor:
        input = self.flatten(input)
        return self.nn(input)
    
    def loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return self.criterion(outputs, targets)
    
    def fit(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return argmax(outputs, dim=1), loss
    
    def evaluate(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]: 
        outputs = self(inputs)
        return argmax(outputs, dim=1), self.loss(outputs, targets)