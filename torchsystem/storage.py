from os import path
from typing import Optional
from typing import Any
from logging import getLogger
from mlregistry import Registry
from mlregistry import get_hash, get_metadata
from abc import ABC
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchsystem.settings import Settings
from torchsystem.weights import Weights
from torchsystem.aggregate import Aggregate

logger = getLogger(__name__)

class Storage[T]:
    '''
    Base class for storage classes. It is responsible for coordinating the registry and weights of objects. 
    '''
    registry: Registry[T]
    weights: Weights[T]
    category: str
    
    @classmethod
    def register(cls, type: type):
        logger.info(f'Registering {type.__name__} in category {cls.category}')
        cls.registry.register(type, cls.category)
        
    def get(self, name: str, *args, **kwargs) -> Optional[T]:
        '''
        Get an object from the registry and restore it's weights if available.

        Parameters:
            name (str): The name of the object.
            *args: Positional arguments for initializing the object.
            **kwargs: Keyword arguments for initializing the object.
        '''

        if not name in self.registry.keys():
            return None
        object = self.registry.get(name)(*args, **kwargs)
        if hasattr(self, 'weights'):
            self.weights.restore(object, f'{self.category}:{get_hash(object)}')
        return object
    
    def store(self, object: T):
        '''
        Store the weights of an object in a given category, a warning is logged if the object is not registered.

        Parameters:
            object (T): The object to store.
        '''
        logger.info(f'Storing {object.__class__.__name__} in category {self.category}')
        assert object.__class__.__name__ in self.registry.keys(), f'{object.__class__.__name__} not registered in {self.category}'
        if hasattr(self, 'weights'):
            self.weights.store(object, f'{self.category}:{get_hash(object)}')

    def restore(self, object: T):
        '''
        Restore the weights of an object from a given category, a warning is logged if the object is not registered.

        Parameters:
            object (T): The object to restore.
        '''
        logger.info(f'Restoring {object.__class__.__name__} in category {self.category}')
        assert object.__class__.__name__ in self.registry.keys(), f'{object.__class__.__name__} not registered in {self.category}'
        if hasattr(self, 'weights'):
            self.weights.restore(object, f'{self.category}:{get_hash(object)}')

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
