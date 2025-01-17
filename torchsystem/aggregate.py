from abc import ABC, abstractmethod
from typing import Any
from typing import Literal
from torch.nn import Module
from pymsgbus import Producer, Consumer
from pymsgbus.models import Event

class Aggregate(Module, ABC):
    """
    An AGGREGATE is a cluster of associated objects that we treat as a unit for the purpose
    of data changes. Each AGGREGATE has a root and a boundary. The boundary defines what is
    inside the AGGREGATE. The root is a single, specific ENTITY contained in the AGGREGATE.

    An AGGREGATE is responsible for maintaining the consistency of the data within its boundary
    and enforcing invariants that apply to the AGGREGATE as a whole. It can communicate data
    to the outside world and execute complex logic using domain events.

    In deep learning, an AGGREGATE consist not only of a neural network, but also several other
    components such as optimizers, schedulers, tokenizers, etc. In order to perform complex tasks.

    For example, a transformer model is just a neural network, and in order to perform tasks such
    as text completion or translation, it needs to be part of an AGGREGATE that includes other 
    components like a tokenizer. The AGGREGATE is responsible for coordinating the interactions   
    between these components.

    Attributes:
        id (Any): The id of the AGGREGATE ROOT. It should be unique within the AGGREGATE boundary.
        phase (Literal['train', 'evaluation']): The phase of the AGGREGATE.

    Methods:
        emit:
            Emits an event to all consumers of the AGGREGATE.

        register:
            Bind a group of consumers to the AGGREGATEs producer.

    Example:

        .. code-block:: python

        from torch import Tensor
        from torch.nn import Module
        from torch.optim import Optimizer
        from torchsystem import Aggregate

        class Classifier(Aggregate):
            def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
                super().__init__()
                self.epoch = 0
                self.model = model
                self.criterion = criterion
                self.optimizer = optimizer

            def forward(self, input: Tensor) -> Tensor:
                return self.model(input)
            
            def loss(self, output: Tensor, target: Tensor) -> Tensor:
                return self.criterion(output, target)

            def fit(self, input: Tensor, target: Tensor) -> tuple[Tensor, float]:
                self.optimizer.zero_grad()
                output = self(input)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                return output, loss.item()

            def evaluate(self, input: Tensor, target: Tensor) -> tuple[Tensor, float]: 
                output = self(input)
                loss = self.loss(output, target)
                return output, loss.item()
    """
    def __init__(self):
        super().__init__()
        self.producer = Producer()

    @property
    def id(self) -> Any:
        """
        The id of the AGGREGATE ROOT. It should be unique within the AGGREGATE boundary.
        """
        raise NotImplementedError("The id property must be implemented.")
        
    @property
    def phase(self) -> Literal['train', 'evaluation']:
        return 'train' if self.training else 'evaluation'
    
    @phase.setter
    def phase(self, value: Literal['train', 'evaluation']):
        self.train() if value == 'train' else self.eval()

    def publish(self, event: Event):
        """
        Emit an event to all consumers of the AGGREGATE. The event is put on an event queue
        and triggers the execution of a chain of event with handlers for all events that
        are enqueued or being emitted in this process.

        Args:
            event (Any): The event to emit.

        Example:

            .. code-block:: python
            from dataclasses import dataclass
            from torchsystem import Consumer

            @dataclass
            class SomeEvent:
                message: str

            consumer = Consumer()

            @consumer.handler
            def on_event(event: SomeEvent):
                print(f"Event message: {event.message}")

            classifier.bind(consumer)
            classifier.publish(SomeEvent(message="Hello World!"))
        """
        self.producer.emit(event)


    def bind(self, *consumers: Consumer):
        """
        Bind a group of observers to the AGGREGATE. Each observer will consumer EVENTS
        from the AGGREGATE.

        Args:
            consumers (Consumer): The consumers to bind.
        """
        for consumer in consumers:
            assert isinstance(consumer, Consumer), f"Consumer {consumer} is not a valid observer type."
            self.producer.register(consumer)

    def fit(self, *args, **kwargs) -> Any:
        """
        Fit the defined model to the given data. This method should be overriden by the user. 

        
        Example:

            .. code-block:: python

            class Classifier(Aggregate):
                def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
                    super().__init__()
                    self.model = model
                    self.criterion = criterion
                    self.optimizer = optimizer
                
                def loss(self, output: Tensor, target: Tensor) -> Tensor:
                    return self.criterion(output, target)

                def fit(self, input: Tensor, target: Tensor) -> tuple[Tensor, float]:
                    self.optimizer.zero_grad()
                    output = self(input)
                    loss = self.loss(output, target)
                    loss.backward()
                    self.optimizer.step()
                    return output, loss.item()



        """
        raise NotImplementedError("The fit method must be implemented.")
    
    def evaluate(self, *args, **kwargs) -> Any:
        """
        Evaluate the defined model on the given data. This method should be overriden by the user.

        
        Example:

            .. code-block:: python

            class Classifier(Aggregate):
                def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
                    super().__init__()
                    self.model = model
                    self.criterion = criterion
                    self.optimizer = optimizer
                    
                def loss(self, output: Tensor, target: Tensor) -> Tensor:
                    return self.criterion(output, target)

                def evaluate(self, input: Tensor, target: Tensor) -> tuple[Tensor, float]: 
                    output = self(input)
                    loss = self.loss(output, target)
                    return output, loss.item()
        """
        raise NotImplementedError("The evaluate method must be implemented.")
    
    def predict(self, *args, **kwargs) -> Any:
        """
        Predict the output of the model on the given data. This method should be overriden by the user.

        
        Example:

            .. code-block:: python
            from torch import argmax

            class Classifier(Aggregate):
                def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
                    super().__init__()
                    self.model = model
                    self.criterion = criterion
                    self.optimizer = optimizer

                def predict(self, input: Tensor) -> Tensor:
                    return argmax(self(input), dim=1)
        """
        raise NotImplementedError("The predict method must be implemented.")