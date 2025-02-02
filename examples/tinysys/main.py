from logging import basicConfig, INFO

from torch import cuda
from torch.nn import ReLU
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsystem.registry import register 

from modules.mlp import MLP
from datasets.mnist import Digits 
from tinysys.metrics import Metrics 
from tinysys.repository import Repository
from tinysys.adapters import getmodels
from tinysys.services import (
    training,
    logging,
    compilation,
    storage,
    tensorboard,
)

EXPERIMENT = 'digits-mnist-mlp'

basicConfig(level=INFO)
register(MLP)
register(ReLU)
register(Digits)
register(CrossEntropyLoss)
register(Adam, excluded_args=[0])
register(DataLoader)
 
summarywriter = SummaryWriter(f'data/logs/{EXPERIMENT}')

def device():
    return 'cuda' if cuda.is_available() else 'cpu'

def models():
    return getmodels(EXPERIMENT)

def repository():
    return Repository(path=EXPERIMENT)

def writer():
    yield summarywriter
    summarywriter.flush()

training.provider.override(training.device, device)
training.producer.register(logging.consumer)
training.producer.register(storage.consumer)
training.producer.register(tensorboard.consumer)
storage.provider.override(storage.models, models)
storage.provider.override(storage.repository, repository)
compilation.provider.override(compilation.device, device)
compilation.provider.override(compilation.models, models)
compilation.provider.override(compilation.repository, repository)
tensorboard.provider.override(tensorboard.writer, writer)

nn = MLP(784, 256, 10, dropout=0.1, activation=ReLU())
criterion = CrossEntropyLoss()
optimizer = Adam(nn.parameters(), lr=1e-3)
metrics = Metrics(device = device())
loaders = [
    ('train', DataLoader(Digits(train=True, normalize=True), batch_size=32, shuffle=True)),
    ('validation', DataLoader(Digits(train=False, normalize=True), batch_size=32, shuffle=False))
] 
classifier = compilation.compiler.compile(nn, criterion, optimizer)

for epoch in range(10):
    training.service.handle('iterate', classifier, loaders, metrics)
 
summarywriter.close()