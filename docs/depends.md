### Inject Dependencies in your services.

Dependency Injection (DI) is a design pattern in programming that allows components or functions to declare their dependencies rather than creating them internally. This makes the code more modular, testable, and maintainable.

This framework provides a simple but powerful dependency injection system inspired by the used in the FastAPI library. It works with function-based dependency resolution and handles both standard and generator-based dependencies. It also supports overriding dependencies, allowing you to define your logic using interfaces and override them with concrete implementations later in the code.

Example:

```python
# src/services/training.py
from torch import cuda
from torchsystem import Depends
from torchsystem.services import Service
from sqlalchemy.orm import Session

service = Service() # You will know more about services in the next sections.
                    # For now, just think of it as a decorator that allows you
                    # to inject dependencies in your functions.

def device() -> str:
    return 'cuda' if cuda.is_available() else 'cpu'

def trainer(): 
    raise NotImplementedError('Trainer not implemented')

def db_session() -> Session:
    ...

@service.hander
def train(model, trainer=Depends(trainer), device=Depends(device), db=Depends(db_session)):
    ...
```

In the example above, the `train` function has three dependencies: `trainer`, `device`, and `db`. The `Depends` class is used to declare these dependencies. When the `train` function is called, the dependencies are resolved and passed to the function automatically. As you may have noticed, the last two dependencies are not implemented, they are dependencies that can be overrided when plugging the infrastructure in the application layer.

```python	
# src/app.py
from src.services import training

def trainer():
    return MaybeLightningTrainer()# Maybe you want to use PyTorch Lightning, all it's up to you.

def db_session():
    session = sessionmaker()
    session.begin()
    try:
        yield session
    finally:
        session.close() # Clean up resources

training.service.dependency_overrides[training.trainer] = trainer
training.service.dependency_overrides[training.db_session] = db_session
```

In the example above, we override the `trainer` and `db_session` dependencies with concrete implementations. Note that the `db_session` dependency is a generator-based dependency, which allows you to clean up resources after the dependency is used. There are a lot of situations where you need to clean up resources after using a dependency, for example when working with distributed training, databases, filesystems, tensorboard, etc.

You will see more examples of dependency injection in the next sections, dependency injection is a core concept in this framework and is used extensively in the compiler, services, consumers or subscribers. 