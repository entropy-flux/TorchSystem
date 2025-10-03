from torch import Tensor
from torch import argmax
from torch.nn import Module
from torch.optim import Optimizer
from torchsystem import Aggregate
from torchsystem.registry import gethash, getname

class Classifier(Aggregate):
    def __init__(self, nn: Module, criterion: Module, optimizer: Optimizer):
        super().__init__()
        self.epoch = 0
        self.nn = nn
        self.criterion = criterion
        self.optimizer = optimizer
        self.name = getname(nn)
        self.hash = gethash(nn)

    def forward(self, input: Tensor) -> Tensor:
        return self.nn(input)
     
    def loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return self.criterion(outputs, targets)

    def fit(self, inputs: Tensor, targets: Tensor) -> Tensor:
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def evaluate(self, inputs: Tensor, targets: Tensor) -> Tensor: 
        outputs = self(inputs)
        return self.loss(outputs, targets)