from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST

from torchsystem.settings import Settings
from torchsystem import Session
from torchsystem.metrics import Callbacks
from torchsystem.metrics.average import Loss, Accuracy

from examples.simple_example.aggregate import Classifier, ClassifierSettigs
from examples.simple_example.repository import Classifiers
from examples.simple_example.model import MLP
from examples.simple_example.services import service

settings = Settings(aggregate=ClassifierSettigs(device='cpu'))
model = MLP(input_size=784, hidden_size=128, output_size=10, dropout=0.5)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
classifier = Classifier('1', model, criterion, optimizer, settings)
repository = Classifiers(settings)
callback = Callbacks(Loss(), Accuracy())
dataloaders = [
    ('train', DataLoader(MNIST(root='.', train=True, download=True))),
    ('test', DataLoader(MNIST(root='.', train=False, download=True)))
]

with Session(repository) as session:
    session.on(StopIteration)(lambda: session.commit())
    repository.put(classifier)
    for epoch in range(10):
        for phase, loader in dataloaders:
            classifier.phase = phase
            classifier.fit(loader, callback) if phase == 'train' else classifier.evaluate(loader, callback)