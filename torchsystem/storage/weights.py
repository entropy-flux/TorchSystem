import os
from logging import getLogger
from typing import Optional
from torch import save, load
from torch.nn import Module

logger = getLogger(__name__)

class Weights[T: Module]:
    """
    A class for storing and restoring the weights of a module. The Weights class is used by the Storage class
    to store and restore the weights of a module. The Weights class is a simple wrapper around the torch.save
    and torch.load functions and provides an OO interface to be used as a data access object in the implementation
    of repositories.

    Attributes:
        location (str): The location of the weights.
        extension (str): The file extension of the weights.

    Methods:

        store:
            Store the weights of a module.

        restore:    
            Restore the weights of a module.

    Example:

        .. code-block:: python

        from torchsystem.storage import Repository, Weights

        class Models(Repository[Module]):
            def __init__(self):
                self.weights = Weights(root='data/weights', path='models')

            def store(self, module: Module):
                self.weights.store(module)

            def restore(self, module: Module):
                self.weights.restore(module)
    """
    def __init__(
        self, 
        root: str = 'data/weights', 
        path: str = None,
        extension: str = '.pth'
    ):
        """
        Initialize the Weights class.

        Args:
            root (str, optional): The root directory to store the weights.
            path (str, optional): The path to store the weights.
            extension (str, optional): The file extension to store the weights.
        """
        self.location = os.path.join(root, path) if path else root
        self.extension = extension
        if not os.path.exists(self.location):
            os.makedirs(self.location)
    
    def store(self, module: T, filename: Optional[str] = None):
        """
        Store the weights of a module. The weights are stored in the location with the filename and extension.
        If the filename is not provided, the class name of the module is used as the filename.
        Args:
            module (T): The module to store its weights.
            filename (Optional[str], optional): The filename to store the weights. Defaults to None.
        """
        filename = filename or module.__class__.__name__
        save(module.state_dict(), os.path.join(self.location, filename + self.extension))

    def restore(self, module: T, filename: Optional[str]) -> bool:
        """
        Restore the weights of a module. The weights are restored from the location with the filename and extension.

        Args:
            module (T): The module to restore its weights.
            filename (Optional[str]): The filename to restore the weights.

        Returns:
            bool: True if the weights were restored, False otherwise.
        """
        filename = filename or module.__class__.__name__
        try:
            state_dict = load(os.path.join(self.location, filename + self.extension), weights_only=True)
            module.load_state_dict(state_dict)
            return True
        except FileNotFoundError:
            return False