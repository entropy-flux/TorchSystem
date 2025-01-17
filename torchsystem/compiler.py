from typing import Callable
from typing import Any
from torch import compile
from pymsgbus.depends import inject
from pymsgbus.depends import Provider
from torchsystem.settings import Settings
from torchsystem.aggregate import Aggregate

class Compiler[T: Aggregate]:
    """
    In DDD, AGGREGATEs usually have a complex structure and are built from multiple components. The
    process of building an AGGREGATE can be broken down into multiple steps. In the context of
    neural networks, AGGREGATEs not only should be built but also compiled. Compilation is the
    process of converting a high-level neural network model into a low-level representation that can
    be executed on a specific hardware platform and can be seen as an integral part of the process
    of building an AGGREGATE.

    A `Compiler` is a class that compiles a pipeline of functions to be executed in sequence in order
    to build an AGGREGATE and has the ability to compile the AGGREGATE to a low-level representation for
    execution on a specific hardware platform. It acts as the builder pattern for AGGREGATEs.

    Attributes:
        pipeline (list[Callable[..., Any]]): A list of functions to be executed in sequence.
        settings (Settings): The settings used by the compiler. Defaults to the pytorch's default settings.

    Methods:
        compile:
            Compiles and AGGREGATE to a low-level representation for execution on a specific hardware platform
            using the settings provided to the compiler.

        step:
            Add a function to the compilation pipeline.

    Example:

        .. code-block:: python
        from logging import getLogger
        from torch import cuda
        
        compiler = Compiler[Classifier]()
        logger = getLogger(__name__)

        @compiler.step
        def build_classifier(model, criterion, optimizer):
            logger.info(f'Building classifier')
            logger.info(f'- model: {model.__class__.__name__}')
            logger.info(f'- criterion: {criterion.__class__.__name__}')
            logger.info(f'- optimizer: {optimizer.__class__.__name__}')
            return Classifier(model, criterion, optimizer)

        @compiler.step
        def move_to_device(classifier: Classifier):
            device = 'cuda' if cuda.is_available() else 'cpu'
            logger.info(f'Moving classifier to device: {device}')
            return classifier.to(device)

        @compiler.step
        def compile_classifier(classifier: Classifier):
            logger.info(f'Compiling classifier')
            return compiler.compile(classifier)
        ...

        classifier = compiler(model, criterion, optimizer)
    """

    def __init__(self, settings: Settings = None, provider: Provider = None, cast: bool = False):
        self.provider = provider or Provider()
        self.pipeline = list[Callable[..., Any]]()
        self.cast = cast
        """
        Initialize the Compiler with the given settings. If no settings are provided, the default
        settings are used, wich are the same as the pytorch's default settings for compilation.

        Args:
            settings (Settings | None, optional): The settings used by the compiler. Defaults to None.
        """
        self.settings = settings or Settings()

    def compile(self, compilable: Any) -> Any:
        """
        Compiles an AGGREGATE to a low-level representation for execution on a specific hardware platform
        using the CompilerSettings or pytorch's default settings for compilation.

        Args:
            compilable (T): The module or function to be compiled.

        Returns:
            T: The compiled module or function.
        """
        return compile(
            compilable,
            fullgraph=self.settings.compiler.fullgraph,
            dynamic=self.settings.compiler.dynamic,
            backend=self.settings.compiler.backend,
            mode=self.settings.compiler.mode,
            disable=self.settings.compiler.disable
        )

    def step(self, callable: Callable[..., Any]) -> Callable[..., Any]:
        """
        A decorator that adds a function to the compilation pipeline. The function will be executed in
        sequence when the pipeline is compiled. 

        Args:
            callable (Callable[..., Any]): The function to be added to the compilation pipeline.

        Returns:
            Callable[..., Any]: The function added to the compilation pipeline.

        Example:

        .. code-block:: python
        
        from torchsystem import Depends
        from torchsystem.compiler import Compiler

        compiler = Compiler()

        ...
        def device():
            return 'cuda' if cuda.is_available() else 'cpu
        
        @compiler.step
        def build_classifier(model, criterion, optimizer, device=Depends(device)):
            logger.info(f'Building classifier')
            classifier = Classifier(model, criterion, optimizer).to(device)
            return compiler.compile(classifier)
        """
        injected = inject(dependency_overrides_provider=self.provider, cast=self.cast)(callable)
        self.pipeline.append(injected)
        return injected
    
    def __call__(self, *args, **kwargs) -> T:
        """
        Executes the compilation pipeline in sequence. The output of each function is passed as input to
        the next function in the pipeline. The first function in the pipeline receives the arguments
        passed to the compiler.

        The result of the last function in the pipeline is returned and should be the compiled AGGREGATE.
        
        Returns:
            T: The compiled AGGREGATE.
        """
        result = None
        for step in self.pipeline:
            result = step(*args, **kwargs) if not result else step(result) #TODO: This doesn't work with Depends.
        return result