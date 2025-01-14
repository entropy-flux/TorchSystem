from os import makedirs
from os import path, makedirs
from logging import getLogger
from typing import Optional
from torch import save, load
from torch.nn import Module
from torchsystem.settings import Settings

logger = getLogger(__name__)

class Weights[T: Module]:
    '''
    Weights class is responsible for storing and restoring the weights of a module.

    Args:
        directory (str): The directory to store the weights.    
    '''
    def __init__(self, settings: Settings):
        self.settings = settings
        folder = self.settings.storage.folder
        self.location = path.join(self.settings.storage.weights.directory, folder) if folder else self.settings.storage.weights.directory
        if not path.exists(self.location):
            makedirs(self.location)

    
    def store(self, module: T, filename: Optional[str] = None, extension: str = '.pth'):
        '''
        Store the weights of a module.

        Args:
            module (Module): The module to store the weights.
            filename (Optional[str]): The filename to store the
        '''
        if not filename:
            filename = module.__class__.__name__
        logger.info(f'Storing weights of {module.__class__.__name__} in {filename}.pth')	
        save(module.state_dict(), path.join(self.location, filename + extension))
        logger.info(f'Weights stored successfully')
        

    def restore(self, module: T, filename: Optional[str], extension: str = '.pth'):
        '''
        Restore the weights of a module.

        Args:
            module (Module): The module to restore the weights.
            filename (str): The filename to restore the weights.
        '''
        if not filename:
            filename = module.__class__.__name__
        logger.info(f'Restoring weights of {module.__class__.__name__} from {filename}.pth')
        try:
            state_dict = load(path.join(self.location, filename + extension), weights_only=True)
            module.load_state_dict(state_dict)
            logger.info(f'Weights restored successfully')
        except FileNotFoundError as error:
            logger.warning(f'Error restoring weights: {error}')