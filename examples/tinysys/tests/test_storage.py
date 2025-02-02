from dataclasses import dataclass
from torch import Tensor
from torch.nn import Module
from torch.nn import GELU
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsystem.registry import register, gethash

from tinysys.domain import Repository
from tinysys.services import storage 
from tinysys.ports.models import Models
from tinysys.services.training import Trained, Validated, Iterated

from modules.mlp import MLP
from datasets.mnist import Digits
 
class Classifier:
    def __init__(self, nn: Module, criterion: Module, optimizer: Module):
        self.id = gethash(nn)
        self.nn = nn
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 10
        self.phase = 'train'

register(MLP)
register(Digits)
register(CrossEntropyLoss)
register(Adam, excluded_args=[0])
register(GELU)
register(DataLoader) 

def test_publishing(models: Models, repository: Repository):
    storage.consumer.dependency_overrides[storage.models] = lambda: models 
    storage.consumer.dependency_overrides[storage.repository] = lambda: repository

    nn = MLP(784, 128, 10, 0.2, activation = GELU())
    criterion = CrossEntropyLoss()
    optimizer = Adam(nn.parameters(), lr=0.001)
    model = Classifier(nn, criterion, optimizer)
    loaders = [
        ('train', DataLoader(Digits(train=True, normalize=True), batch_size=32, shuffle=True)),
        ('valid', DataLoader(Digits(train=False, normalize=True), batch_size=32, shuffle=False))
    ]

    trained = Trained(model, {'loss': Tensor([0.5]), 'accuracy': Tensor([0.9])})
    validated = Validated(model, {'loss': Tensor([0.5]), 'accuracy': Tensor([0.9])})


    models.create(id=model.id, name='test')

    for phase, loader in loaders:
        if phase == 'train':
            model.phase = 'train'
            storage.consumer.consume(trained)
        else:
            model.phase = 'evaluation'
            storage.consumer.consume(validated)
    storage.consumer.consume(Iterated(model, loaders))

    model = models.read(id=model.id)
    assert len(model.modules.list('nn')) == 1
    assert len(model.modules.list('criterion')) == 1
    assert len(model.modules.list('optimizer')) == 1
    assert len(model.metrics.list()) == 4
    assert len(model.iterations.list()) == 2