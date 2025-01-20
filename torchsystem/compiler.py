from typing import Callable
from typing import Any
from torch import compile as compile
from pymsgbus.depends import Depends as Depends
from pymsgbus.depends import inject
from pymsgbus.depends import Provider
from torchsystem.aggregate import Aggregate as Compiled

class Compiler[T: Compiled]:
    """
    AGGREGATES usually have a complex initialization and are built from multiple components. The
    process of building an AGGREGATE can be broken down into multiple steps. In the context of
    neural networks, AGGREGATEs not only should be built but also compiled. Compilation is the
    process of converting a high-level neural network model into a low-level representation that can
    be executed on a specific hardware platform and can be seen as an integral part of the process
    of building an AGGREGATE.

    A `Compiler` is a class that compiles a pipeline of functions to be executed in sequence in order
    to build an a low-level representation of the AGGREGATE. Since some compilation steps sometimes
    requires runtime information, the `Compiler` provides a mechanism to inject dependencies into
    the pipeline.

    Attributes:
        pipeline (list[Callable[..., Any]]): A list of functions to be executed in sequence.

    Methods:
        compile:
            Execute the pipeline of functions in sequence. The output of each function is passed as
            input to the next function. The compiled AGGREGATE should be returned by the last function
            in the pipeline.

        step:
            A decorator that adds a function to the pipeline. The function should take as input the
            output of the previous function in the pipeline and return the input of the next function
            in the pipeline.

    Example:

        .. code-block:: python
        from logging import getLogger
        from torch import cuda
        from torchsystem.compiler import Compiler
        from torchsystem.compiler import Depends
        from torchsystem.compiler import compiler

        compiler = Compiler[Classifier]()
        logger = getLogger(__name__)

        def device():
            raise NotImplementedError

        @compiler.step
        def build_classifier(model, criterion, optimizer):
            logger.info(f'Building classifier')
            logger.info(f'- model: {model.__class__.__name__}')
            logger.info(f'- criterion: {criterion.__class__.__name__}')
            logger.info(f'- optimizer: {optimizer.__class__.__name__}')
            return Classifier(model, criterion, optimizer)

        @compiler.step
        def move_to_device(classifier: Classifier, device = Depends(device)):
            logger.info(f'Moving classifier to device: {device}')
            return classifier.to(device)

        @compiler.step
        def compile_classifier(classifier: Classifier):
            logger.info(f'Compiling classifier')
            return compile(classifier)
        ...

        compiler.dependency_overrides[device] = lambda: 'cuda' if cuda.is_available() else 'cpu'
        classifier = compiler.compie(model, criterion, optimizer)
    """
    def __init__(
        self,
        provider: Provider = None,
        cast: bool = True
    ):
        """
        Initialize the Compiler.

        Args:
            provider (Provider, optional): The dependency provider. Defaults to None.
            cast (bool, optional): Whether to cast the dependencies during injection. Defaults to True.
        """
        self.pipeline = list[Callable]()
        self.provider = provider or Provider()
        self.cast = cast
    
    @property
    def dependency_overrides(self) -> dict:
        return self.provider.dependency_overrides

    def step(self, callable: Callable) -> Any:
        """
        Add a function to the pipeline. The function should take as input the output of the previous
        function in the pipeline and return the input of the next function in the pipeline.

        Args:
            callable (Callable): The function to be added to the pipeline.

        Returns:
            Any: The requirements for the next step in the pipeline.
        """
        injected = inject(callable, dependency_overrides_provider=self.provider, cast=self.cast)
        self.pipeline.append(injected)
        return self
    
    def compile(self, *args, **kwargs) -> T:
        """
        Execute the pipeline of functions in sequence. The output of each function is passed as input
        to the next function. The compiled AGGREGATE should be returned by the last function in the pipeline.
        
        Returns:
            T: The compiled AGGREGATE.
        """
        result = None
        for step in self.pipeline:
            if not result:
                result = step(*args, **kwargs)
            else:
                result = step(*result) if isinstance(result, tuple) else step(result)
        return result