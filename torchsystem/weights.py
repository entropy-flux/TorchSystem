from os import makedirs
from os import path, makedirs
from logging import getLogger
from torch import save, load
from torch.nn import Module

logger = getLogger(__name__)

class Weights[T: Module]:
    '''
    Weights class is responsible for storing and restoring the weights of a module.

    Args:
        directory (str): The directory to store the weights.    
    '''
    def __init__(self, directory: str):
        self.location = directory
        if not path.exists(self.location):
            makedirs(self.location)

    
    def store(self, module: T, filename: str):
        '''
        Store the weights of a module.

        Parameters:
            module (Module): The module to store the weights.
            folder (str): The folder to store the weights.
            filename (str): The filename to store the
        '''
        logger.info(f'Storing weights of {module.__class__.__name__} in {filename}.pth')	
        save(module.state_dict(), path.join(self.location, filename + '.pth'))
        logger.info(f'Weights stored successfully')
        

    def restore(self, module: T, filename: str):
        '''
        Restore the weights of a module.

        Parameters:
            module (Module): The module to restore the weights.
            folder (str): The folder to restore the weights.
            filename (str): The filename to restore the weights.
        '''
        logger.info(f'Restoring weights of {module.__class__.__name__} from {filename}.pth')
        try:
            state_dict = load(path.join(self.location, filename + '.pth'), weights_only=True)
            module.load_state_dict(state_dict)
            logger.info(f'Weights restored successfully')
        except FileNotFoundError as error:
            logger.warning(f'Error restoring weights: {error}')