from torch import compile
from pybondi.aggregate import Factory
from torchsystem.aggregate import Aggregate
from torchsystem.settings import Settings
from logging import getLogger

logger = getLogger(__name__)

class Compiler[T: Aggregate]:
    '''
    The Compiler class is used to compile several modules into a single aggregate.    

    Parameters:
        factory (Callable): A callable that returns an aggregate. Could be the aggregate's type
        a function that initializes and returns an aggregate or a callable instance of a factory.
    '''

    def __init__(self, factory: Factory, settings: Settings = None):
        self.settings = settings or Settings()
        self.factory = factory

    def compile(self, *args, **kwargs) -> T:
        '''
        Builds and compiles the aggregate using the factory provided.

        Parameters:
            *args: The positional arguments to pass to the factory.
            **kwargs: The keyword arguments to pass to the factory.                    
        '''
        logger.info(f'Building and compiling the aggregate')
        aggregate = self.factory(*args, **kwargs)
        try:
            logger.info(f'Compiling the aggregate with settings:')
            logger.info(f'-Fullgraph: {self.settings.compilation.fullgraph}')
            logger.info(f'-Dynamic: {self.settings.compilation.dynamic}')
            logger.info(f'-Backend: {self.settings.compilation.backend}')
            logger.info(f'-Mode: {self.settings.compilation.mode}')
            logger.info(f'-Options: {self.settings.compilation.options}')
            logger.info(f'-Disable: {self.settings.compilation.disable}')
            logger.info(f'-Raise on error: {self.settings.compilation.raise_on_error}')
            compiled = compile(aggregate,
                fullgraph=self.settings.compilation.fullgraph,
                dynamic=self.settings.compilation.dynamic,
                backend=self.settings.compilation.backend,
                mode=self.settings.compilation.mode,
                options=self.settings.compilation.options,
                disable=self.settings.compilation.disable,
            )
            logger.info(f'Aggregate compiled successfully')
            logger.info(f'Aggregate: {compiled}')
            return compiled
        except Exception as error:
            logger.error(f'Error compiling the aggregate: {error}')
            if self.settings.compilation.raise_on_error:
                raise error
            logger.info(f'Returning the uncompiled aggregate')
            return aggregate