from typing import Optional
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset
from mlregistry import Registry
from mlregistry import gethash
from torchsystem.storage.weights import Weights

class Storage[T]:
    """
    A class for storing and restoring objects from the registry. The Storage class coordinates the
    storage and registry of objects. The registry is a data structure that maps names to objects facilitating
    the retrieval of torch objects using serializable has a set of other functionalities. See them in the
    [mlregistry documentation](https://mr-mapache.github.io/ml-registry/). The Storage class also provides a mechanism
    to store and restore the weights of the objects.

    Attributes:
        weights (Optional[Weights[T]]): The weights of the objects if any.
        registry (Registry[T]): The registry of the objects.
        category (str): The category of the objects.
    """
    weights: Optional[Weights[T]]
    registry: Registry[T]
    category: str
    
    @classmethod
    def register(cls, type: type):
        """
        Register a type in the registry. The registry holds a mapping between names and types and provides
        a mechanism to retrieve objects from the registry.

        Args:
            type (type): _description_

        Returns:
            _type_: _description_
        """
        cls.registry.register(type, cls.category)
        return type

    @classmethod
    def build(cls, name: str, *args, **kwargs) -> Optional[T]:
        '''
        Build an object from the registry. The build method is a factory method that creates an object from the
        registry using the name of the object.

        Args:
            name (str): The name of the object.
            *args: Positional arguments for initializing the object.
            **kwargs: Keyword arguments for initializing the object.

        Returns:
            Optional[T]: The object from the registry.
        '''
        if not name in cls.registry.keys():
            return None
        object = cls.registry.get(name)(*args, **kwargs)
        return object
        
    def get(self, name: str, *args, **kwargs) -> Optional[T]:
        """
        Get an object from the registry and restore its weights.

        Args:
            name (str): The name of the object.
            *args: Positional arguments for initializing the object.
            **kwargs: Keyword arguments for initializing the object.

        Returns:
            Optional[T]: The object from the registry with restored weights if available.
        """
        if not name in self.registry.keys():
            return None
        object = self.registry.get(name)(*args, **kwargs)
        if hasattr(self, 'weights'):
            self.weights.restore(object, f'{self.category}-{object.__class__.__name__}-{gethash(object)}' )
        return object
    
    def store(self, object: T):
        """
        Store the object's weights if available.

        Args:
            object (T): The object to store its weights.
        """
        assert object.__class__.__name__ in self.registry.keys(), f'{object.__class__.__name__} not registered in {self.category}'
        if hasattr(self, 'weights'):
            self.weights.restore(object, f'{self.category}-{object.__class__.__name__}-{gethash(object)}' )

    def restore(self, object: T):
        """
        Restore the object's weights if available.

        Args:
            object (T): The object to restore its weights.
        """
        assert object.__class__.__name__ in self.registry.keys(), f'{object.__class__.__name__} not registered in {self.category}'
        if hasattr(self, 'weights'):
            self.weights.restore(object, f'{self.category}-{object.__class__.__name__}-{gethash(object)}' )


class Models(Storage[Module]):
    """
    A subclass of the Storage class for storing and restoring neural network models.
    """
    category = 'model'
    registry = Registry()

    def __init__(
        self, 
        root: str = 'data/weights', 
        path: str = None,
        extension: str = '.pth'
    ):
        self.weights = Weights(root, path, extension)


class Criterions(Storage[Module]):
    """
    A subclass of the Storage class for storing and restoring weights of loss functions. Some loss functions
    may have weights that need to be stored and restored
    """
    category = 'criterion'
    registry = Registry()

    def __init__(
        self, 
        root: str = 'data/weights', 
        path: str = None,
        extension: str = '.pth'
    ):
        self.weights = Weights(root, path, extension)


class Optimizers(Storage[Optimizer]):
    """
    A subclass of the Storage class for storing and restoring optimizers from the registry.
    """
    category = 'optimizer'
    registry = Registry(excluded_positions=[0], exclude_parameters={'params'})

    def __init__(
        self, 
        root: str = 'data/weights', 
        path: str = None,
        extension: str = '.pth'
    ):
        self.weights = Weights(root, path, extension)


class Datasets(Storage[Dataset]):
    """
    A subclass of the Storage class for storing and restoring datasets objects from the registry.

    Args:
        Storage (_type_): _description_
    """
    category = 'dataset'
    registry = Registry(exclude_parameters={'root', 'download'})
