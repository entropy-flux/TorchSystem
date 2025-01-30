from abc import ABC
from typing import Any
from typing import Literal
from torch.nn import Module
from torchsystem.domain import Events

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