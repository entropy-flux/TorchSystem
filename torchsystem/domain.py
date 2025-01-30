from abc import ABC
from typing import Any
from typing import Literal
from torch.nn import Module
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Iterable
from inspect import signature
from collections import deque
from collections.abc import Callable

class Event:
    """
    A DOMAIN EVENT is a representation of something that has happened in the DOMAIN.

    This class is a base class for creating custom DOMAIN EVENTS. It is a simple class that can be
    optionally subclassed to write self-documented code when creating custom events.
    """

type EVENT = Event | type[Event] | Exception | type[Exception]
type HANDLERS = Callable | Sequence[Callable]

class Events:
    """
    A collection of DOMAIN EVENTS that have occurred within a Bounded context. The EVENTS
    class is responsible for managing the events that have occurred within the Bounded context
    and dispatching them to the appropriate handlers.

    When an event is enqueued, it is added to the queue of events to be processed. When the `commit`
    method is called, the events are dequeued and dispatched to the appropriate handlers. If no handler
    is found for an event, the event is ignored, except if the event is an exception.

    Exceptions are treated as domain events but they are raised when the `commit` method is called by
    default if no handler is found for it's type.

    Attributes:
        queue (deque[Event]): A queue of DOMAIN EVENTS that have occurred within the Bounded context.
        handlers (dict[type[Event], Sequence[Callable]]): A dictionary of handlers that are responsible for handling
            DOMAIN EVENTS. The key is the type of the event and the value is the handler function.

    Example:
        ```python
        from torchsystem.domain import Events, Event

        class ClsEvent(Event):...

        class ObjEvent(Event):
            def __init__(self, value):
                self.value = value

        class OtherObjEvent(Event):
            def __init__(self, willbeignored):
                self.value = willbeignored

        events = Events()
        events.enqueue(ClsEvent)
        events.enqueue(KeyError) # Enqueues a KeyError exception event
        events.enqueue(ObjEvent('somevalue'))
        events.enqueue(OtherObjEvent('willbeignored'))
        events.enqueue(StopIteration) # Enqueues a StopIteration exception event

        events.handlers[ClsEvent] = lambda: print('ClsEvent was handled.')
        events.handlers[KeyError] = lambda: print('KeyError was handled.')
        events.handlers[ObjEvent] = lambda event: print(f'ObjEvent was handled with value: {event.value}')
        events.handlers[OtherObjEvent] = lambda: print('OtherObjEvent was handled.')

        try:
            events.commit()
        except StopIteration:
            print('StopIteration exception was raised.')

        # Output:
        #ClsEvent was handled.
        #KeyError was handled.
        #ObjEvent was handled with value: somevalue
        #OtherObjEvent was handled.
        #StopIteration exception was raised. Usefull for early stopping in training loops.
        ```
    """
    def __init__(self):
        self.queue = deque[EVENT]()
        self.handlers = dict[type, HANDLERS]()

    @overload
    def enqueue(self, event: Event) -> None: ...

    @overload
    def enqueue(self, event: type[Event]) -> None: ...

    @overload
    def enqueue(self, event: Exception) -> None: ...

    @overload
    def enqueue(self, event: type[Exception]) -> None: ...

    def enqueue(self, event: EVENT) -> None:
        """
        Enqueue a DOMAIN EVENT into the EVENTS queue to be processed when the `commit`
        method is called. Exceptions can also be enqueued as domain events.

        Args:
            event (Event): The DOMAIN EVENT or exception to be enqueued.
        """
        self.queue.append(event)

    def dequeue(self) -> Optional[EVENT]:
        """
        Dequeue a DOMAIN EVENT from the EVENTS queue to be processed by the `commit` method.

        Returns:
            Optional[Event]: The DOMAIN EVENT or exception to be processed.
        """
        return self.queue.popleft() if self.queue else None

    @overload
    def handle(self, event: Event) -> None: ...

    @overload
    def handle(self, event: type[Event]) -> None: ...

    @overload
    def handle(self, event: Exception) -> None: ...

    @overload
    def handle(self, event: type[Exception]) -> None: ...

    def handle(self, event: EVENT) -> None:
        """
        Handles a DOMAIN EVENT by dispatching it to the appropriate handler or group of handlers. If no handler 
        is found for the event, the event is ignored, except if the event is an exception. If the event is an
        exception, it is raised by default if no handler is found for it's type.

        Both classes and instances of DOMAIN EVENTS are supported. The method also will look at the
        signature of the handler to determine if the event should be passed as an argument to the handler
        or if the handler should be called without arguments.
        
        Args:
            event (Event): The DOMAIN EVENT or exception to be handled.

        Raises:
            event: If no handler is found for the event and the event is an exception.
        """
        handlers = self.handlers.get(event) if isinstance(event, type) else self.handlers.get(type(event)) 
        if handlers:
            for handler in handlers if isinstance(handlers, Iterable) else [handlers]:
                handler() if len(signature(handler).parameters) == 0 else handler(event)
        
        elif isinstance(event, Exception) or issubclass(event, Exception):
            raise event

    def commit(self) -> None:
        while event := self.dequeue():
            self.handle(event)


class Aggregate(Module, ABC):
    """
    An AGGREGATE is a cluster of associated objects that we treat as a unit for the purpose
    of data changes. Each AGGREGATE has a root and a boundary. The boundary defines what is
    inside the AGGREGATE. The root is a single, specific ENTITY contained in the AGGREGATE and
    provides the IDENTITY of the AGGREGATE. The root is the only member of the AGGREGATE that
    outside objects are allowed to hold references to.

    In deep learning, an AGGREGATE consist not only of a neural network, but also several other
    components such as optimizers, schedulers, tokenizers, etc.  For example, a transformer model
    is just a neural network, and in order to perform tasks such as text completion or translation,
    it needs to be part of an AGGREGATE that includes other components like a tokenizer. The AGGREGATE
    is responsible for coordinating the interactions between these components.

    Attributes:
        id (Any): The id of the AGGREGATE ROOT. It should be unique within the AGGREGATE boundary.
        phase (Literal['train', 'evaluation']): The phase of the AGGREGATE.
        events (Events): The domain events of the AGGREGATE.

    Methods:
        onphase:
            A hook that is called when the phase changes. Implement this method to add custom behavior.

        onepoch:
            A hook that is called when the epoch changes. Implement this method to add custom behavior.

    Example:
        ```python	
        from torch import Tensor
        from torch.nn import Module
        from torch.optim import Optimizer
        from torchsystem import Aggregate
        from torchsystem.registry import gethash

        class Classifier(Aggregate):
            def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
                super().__init__()
                self.epoch = 0
                self.model = model
                self.criterion = criterion
                self.optimizer = optimizer

            @property
            def id(self) -> str:
                return gethash(self.model) # See the registry module for more information.

            def onepoch(self):
                print(f'Epoch: {self.epoch}')

            def onphase(self):
                print(f'Phase: {self.phase}')

            def forward(self, input: Tensor) -> Tensor:
                return self.model(input)
            
            def loss(self, output: Tensor, target: Tensor) -> Tensor:
                return self.criterion(output, target)

            def fit(self, input: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
                self.optimizer.zero_grad()
                output = self(input)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                return output, loss

            def evaluate(self, input: Tensor, target: Tensor) -> tuple[Tensor, Tensor]: 
                output = self(input)
                loss = self.loss(output, target)
                return output, loss
        ```
    """
    
    def __init__(self):
        super().__init__()
        self.events = Events()

    @property
    def id(self) -> Any:
        """
        The id of the AGGREGATE ROOT. It should be unique within the AGGREGATE boundary. It's up to the
        user to define the id of the AGGREGATE ROOT and how it should be generated.
          
        The `gethash` function from the `torchsystem.registry` module can usefull for generating unique
        ids from registered pytorch objects.
        """
        raise NotImplementedError("The id property must be implemented.")
        
    @property
    def phase(self) -> Literal['train', 'evaluation']:
        """
        The phase of the AGGREGATE. The phase is a property of neural networks that not only describes
        the current state of the network, but also determines how the network should behave. 
        
        During the training phase, the network stores the gradients of the weights and biases, and uses them
        to update the weights and biases. During the evaluation phase, the network does not store the gradients
        of the weights and biases, and does not update the weights and biases.

        Returns:
            Literal['train', 'evaluation']: The current phase of the AGGREGATE.
        """
        return 'train' if self.training else 'evaluation'
    
    @phase.setter
    def phase(self, value: str):
        """
        Set the phase of the AGGREGATE. When the phase changes, the onphase hook method is called.
        The phase will be set to 'train' if the value is 'train', otherwise it will be set to 'evaluation'.
        
        Changing the phase of the AGGREGATE will set all the modules in the AGGREGATE to the training or
        evaluation mode respectively.

        Args:
            value (str): The phase of the AGGREGATE. It can be either 'train' or 'evaluation'.
        """
        self.train() if value == 'train' else self.eval()
        self.onphase()

    def onphase(self):
        """
        A hook that is called when the phase changes. Implement this method to add custom behavior.
        """

    def onepoch(self):
        """
        A hook that is called when the epoch changes. Implement this method to add custom behavior.
        """

    def __setattr__(self, name, value):
        if name == 'epoch' and hasattr(self, 'epoch'):
            super().__setattr__(name, value)
            self.onepoch()
        else:        
            super().__setattr__(name, value)