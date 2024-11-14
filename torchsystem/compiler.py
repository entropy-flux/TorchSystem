from typing import Callable
from torch import compile
from pybondi.aggregate import Factory
from torchsystem.aggregate import Aggregate
from torchsystem.settings import Settings
from logging import getLogger

logger = getLogger(__name__)

class Compiler:
    '''
    The Compiler class is used to compile several modules into a single aggregate.    

    Parameters:
        factory (Callable): A callable that returns an aggregate. Could be the aggregate's type
        a function that initializes and returns an aggregate or a callable instance of a factory.
    '''

    def __init__(self, factory: Factory, settings: Settings = None):
        self.settings = settings or Settings()
        self.factory = factory

    def compile(self, *args, **kwargs) -> Aggregate:
        '''
        Builds and compiles the aggregate using the factory provided.

        Parameters:
            *args: The positional arguments to pass to the factory.
            **kwargs: The keyword arguments to pass to the factory.
        '''
        logger.info(f'Building and compiling the aggregate')
        aggregate = self.factory(*args, **kwargs)
        try:
            compiled = compile(aggregate)
            logger.info(f'Aggregate compiled successfully')
            logger.info(f'Aggregate: {compiled}')
            return compiled
        except Exception as error:
            logger.error(f'Error compiling the aggregate: {error}')
            if self.settings.compilation.raise_on_error:
                raise error
            logger.info(f'Returning the uncompiled aggregate')
            return aggregate