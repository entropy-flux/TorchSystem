from os import path
from typing import Optional
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchsystem.settings import Settings
from torchsystem.weights import Weights
from mlregistry import Registry
from mlregistry import get_hash

class Storage[T]:
    registry: Registry[T]
    weights: Weights[T]
    category: str
    
    @classmethod
    def register(cls, type: type):
        return cls.registry.register(type, cls.category)
        
    def get(self, name: str, *args, **kwargs) -> Optional[T]:
        if not name in self.registry.keys():
            return None
        object = self.registry.get(name)(*args, **kwargs)
        if hasattr(self, 'weights'):
            self.weights.restore(object, f'{self.category}:{get_hash(object)}')
        return object
    
    def save(self, object: T):
        assert object.__class__.__name__ in self.registry.keys(), f'{object.__class__.__name__} not registered in {self.category}'
        if hasattr(self, 'weights'):
            self.weights.store(object, f'{self.category}:{get_hash(object)}')
    
class Models(Storage[Module]):
    category = 'model'
    registry = Registry()

    def __init__(self, folder: str | None = None, settings: Settings = None):
        self.settings = settings or Settings()
        self.weights = Weights(path.join(self.settings.weights.directory, folder) if folder else self.settings.weights.directory)

class Criterions(Storage[Module]):
    category = 'criterion'
    registry = Registry()

    def __init__(self, folder: str | None = None, settings: Settings = None):
        self.settings = settings or Settings()
        self.weights = Weights(path.join(self.settings.weights.directory, folder) if folder else self.settings.weights.directory)

class Optimizers(Storage[Optimizer]):
    category = 'optimizer'
    registry = Registry(excluded_positions=[0], exclude_parameters={'params'})
    
    def __init__(self, folder: str | None = None, settings: Settings = None):
        self.settings = settings or Settings()
        self.weights = Weights(path.join(self.settings.weights.directory, folder) if folder else self.settings.weights.directory)

class Datasets(Storage[Dataset]):
    category = 'dataset'
    registry = Registry(exclude_parameters={'root', 'download'})