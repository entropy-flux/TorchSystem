from torch import allclose
from torch import Tensor
from torch.nn import Module
from torch.nn import Sequential, Dropout, Linear, ReLU, Flatten
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer, Adam
from torchsystem.aggregate import Aggregate
from torchsystem.storage import gethash
from torchsystem.storage import Models, Criterions, Optimizers, Datasets

class MLP(Module):
    def __init__(self, input_features, hidden_size, output_features):
        super().__init__()
        self.layers = Sequential(
            Flatten(),
            Linear(input_features, hidden_size),
            ReLU(),
            Dropout(0.5),
            Linear(hidden_size, output_features)
        )
    
    def forward(self, features):
        return self.layers(features)
    

class Classifier(Aggregate):
    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    @property
    def id(self) -> str:
        return gethash(self.model)    

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)
    

class Repository:
    def __init__(self, experiment: str, folder: str = 'data/weights'):
        self.models = Models(folder, experiment)
        self.criterions = Criterions(folder, experiment)
        self.optimizers = Optimizers(folder, experiment)
        self.datasets = Datasets()


    def store(self, classifier: Classifier):
        self.models.store(classifier.model)
        
    def restore(self, classifier: Classifier):
        self.models.restore(classifier.model)

def test_storing_and_restoring(directory):
    repository = Repository('tests', 'data/test')
    repository.models.register(MLP)
    repository.optimizers.register(Adam)
    repository.criterions.register(CrossEntropyLoss)
    
    model = MLP(784, 128, 10)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    classifier = Classifier(model, criterion, optimizer)
    
    repository.store(classifier)

    model = repository.models.get('MLP', 784, 128, 10)
    criterion = repository.criterions.get('CrossEntropyLoss')
    optimizer = repository.optimizers.get('Adam', model.parameters(), lr=1e-3)
    classifier = Classifier(model, criterion, optimizer)
    other = Classifier(model, criterion, optimizer)
    repository.restore(other)
    assert allclose(classifier.model.layers[1].weight, other.model.layers[1].weight)