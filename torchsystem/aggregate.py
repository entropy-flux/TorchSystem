from abc import ABC
from typing import Any
from typing import Literal
from torch.nn import Module
from pymsgbus.events import Events
from pymsgbus.pubsub import Publisher, Subscriber
from pymsgbus.exceptions import Exceptions

class Aggregate(Module, ABC):
    """
    An AGGREGATE is a cluster of associated objects that we treat as a unit for the purpose
    of data changes. Each AGGREGATE has a root and a boundary. The boundary defines what is
    inside the AGGREGATE. The root is a single, specific ENTITY contained in the AGGREGATE.

    An AGGREGATE is responsible for maintaining the consistency of the data within its boundary
    and enforcing invariants that apply to the AGGREGATE as a whole. It can communicate data
    to the outside world and execute complex logic using domain events or messages through in 
    memory message patterns.

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
        self.epochs = 0
        self.events = Events()
        self.publisher = Publisher()
        self.exceptions = Exceptions()

    @property
    def id(self) -> Any:
        """
        The id of the AGGREGATE ROOT. It should be unique within the AGGREGATE boundary.
        """
        raise NotImplementedError("The id property must be implemented.")
    
    @property
    def epoch(self) -> int:
        """
        The current epoch of the AGGREGATE. The epoch is a property on a machine learning aggregate
        and it's determined by the epoch of it's AGGREGATE ROOT. Secondary effects can be triggered
        by the epoch change overriding the `onepoch` method.

        Returns:
            int: The current epoch.
        """
        return self.epochs
    
    @epoch.setter
    def epoch(self, value: int):
        with self.exceptions:
            self.epochs = value
            self.onepoch()

    def onepoch(self):
        """
        A hook that is called when the epoch changes. Implement this method to add custom behavior.
        """
        pass
        
    @property
    def phase(self) -> Literal['train', 'evaluation']:
        """
        The phase of the AGGREGATE. The phase is a property of neural networks that not only describes
        the current state of the network, but also determines how the network should behave. During the
        training phase, the network stores the gradients of the weights and biases, and uses them to update
        the weights and biases. During the evaluation phase, the network does not store the gradients of the
        weights and biases, and does not update the weights and biases.

        Returns:
            Literal['train', 'evaluation']: The current phase of the AGGREGATE.
        """
        return 'train' if self.training else 'evaluation'
    
    @phase.setter
    def phase(self, value: Literal['train', 'evaluation']):
        with self.exceptions:
            self.train() if value == 'train' else self.eval()
            self.onphase()

    def onphase(self):
        """
        A hook that is called when the phase changes. Implement this method to add custom behavior.
        """
        pass

    def publish(self, message: Any, topic: str):
        """
        Publish a message to all subscribers of a given topic.

        Args:
            message (Any): The message to publish.
            topic (str): The topic to publish the message.
        """
        with self.exceptions:
            self.publisher.publish(message, topic)


    def register(self, *observers: Subscriber):
        """
        Bind a group of observers to the AGGREGATE. Each observer will consumer EVENTS
        from the AGGREGATE. You can register an observer from here or you can observe this
        AGGREGATE from an observer since observers also implement the logic registration logic.

        Args:
            consumers (Consumer): The consumers to bind.
        """
        for observer in observers:
            self.publisher.register(observer)
