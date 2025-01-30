from torch import Tensor
from torch import argmax
from torch import compile
from torch.nn import Module
from torch.nn import Sequential, Dropout, Linear, ReLU, Flatten
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer, Adam

from torchsystem.domain import Aggregate
from torchsystem.compiler import Depends
from torchsystem.compiler import Compiler

class MLP(Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.5):
        super(MLP, self).__init__()
        self.layers = Sequential(
            Flatten(),
            Linear(input_size, hidden_size),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_size, output_size)
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)
    
class Classifier(Aggregate):
    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        super().__init__()
        self.epoch = 0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        output = self.model(input)
        return argmax(output, dim=1), self.criterion(output)

def device():
    return 'cpu'

def epochs():
    ...

compiler = Compiler[Classifier]()

@compiler.step
def build_model(nn: Module, criterion: Module, optimizer: Optimizer) -> Classifier:
    return Classifier(nn, criterion, optimizer)

@compiler.step
def move_to_device(classifier: Classifier, device: str = Depends(device)) -> Classifier:
    return classifier.to(device)

@compiler.step
def compile_model(classifier: Classifier) -> Classifier:
    return compile(classifier.model)

@compiler.step
def retrieve_epoch(classifier: Classifier, epoch: int = Depends(epochs)) -> Classifier:
    classifier.epoch = epoch
    return classifier

def test_compiler():
    nn = MLP(28*28, 128, 10)
    criterion = CrossEntropyLoss()
    optimizer = Adam(nn.parameters(), lr=0.01)
    compiler.dependency_overrides[epochs] = lambda: 10
    classifier = compiler.compile(nn, criterion, optimizer)
    assert classifier.epoch == 10