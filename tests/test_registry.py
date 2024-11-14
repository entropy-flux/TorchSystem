from pytest import fixture
from torch import Tensor
from torch.nn import Module
from torch.nn import ReLU, Linear
from torch.nn import Dropout
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torchsystem.storage import Models, Criterions, Optimizers, Datasets
from mlregistry import get_metadata

class Digits(Dataset):
    def __init__(self, train: bool, normalize: bool):
        transform = Compose([ToTensor()]) if not normalize else Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        self.dataset = MNIST(root='data/datasets', train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

class MLP(Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, p: float):
        super().__init__()
        self.input_layer = Linear(input_size, hidden_size)
        self.activation = ReLU()
        self.dropout = Dropout(p)
        self.output_layer = Linear(hidden_size, output_size)

    def forward(self, sequence: Tensor) -> Tensor:
        sequence = self.input_layer(sequence)
        sequence = self.activation(sequence)
        sequence = self.dropout(sequence)
        sequence = self.output_layer(sequence)
        return sequence

@fixture(scope='session')
def registry():
    Criterions.register(CrossEntropyLoss)
    Optimizers.register(Adam)
    Datasets.register(Digits)
    yield


def test_initialization(registry):
    Models.register(MLP)
    model = MLP(784, 128, 10, p=0.2)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    dataset = Digits(train=True, normalize=True)

    metadata = get_metadata(model)
    assert metadata.name == 'MLP'
    assert metadata.arguments == {'input_size': 784, 'hidden_size': 128, 'output_size': 10, 'p': 0.2}
    assert metadata.type == 'model'
    metadata = get_metadata(criterion)
    assert metadata.arguments == {}
    assert metadata.type == 'criterion'
    metadata = get_metadata(optimizer)
    assert metadata.arguments == {'lr': 0.001}
    assert metadata.type == 'optimizer'
    metadata = get_metadata(dataset)
    assert metadata.arguments == {'train': True, 'normalize': True}


def test_retrieval(registry):
    models = Models()
    criterions = Criterions()
    optimizers = Optimizers()
    datasets = Datasets()
    
    assert 'MLP' in models.registry.keys()
    assert 'CrossEntropyLoss' in criterions.registry.keys()
    assert 'Adam' in optimizers.registry.keys()
    assert 'Digits' in datasets.registry.keys()
        
    model = models.get('MLP',784, 128, 10, p=0.2)
    criterion = criterions.get('CrossEntropyLoss')
    optimizer = optimizers.get('Adam', model.parameters(), lr=0.001)
    dataset = datasets.get('Digits', train=True, normalize=True)

    assert isinstance(model, MLP)
    assert isinstance(criterion, CrossEntropyLoss)
    assert isinstance(optimizer, Adam)
    assert isinstance(dataset, Digits)

    metadata = get_metadata(model)
    assert metadata.name == 'MLP'
    assert metadata.arguments == {'input_size': 784, 'hidden_size': 128, 'output_size': 10, 'p': 0.2}
    assert metadata.type == 'model'
    metadata = get_metadata(criterion)
    assert metadata.arguments == {}
    assert metadata.type == 'criterion'
    metadata = get_metadata(optimizer)
    assert metadata.arguments == {'lr': 0.001}
    assert metadata.type == 'optimizer'
    metadata = get_metadata(dataset)
    assert metadata.arguments == {'train': True, 'normalize': True}  

