### Treat your models as Entities.
 
An ENTITY is an object defined by it's IDENTITY, and it's mutable nature distinguish them from VALUE OBJECTS. In the
context of machine learning, neural networks are stateful objects that can mutate their internal state during training.
This means that they must be treated as entities, and in order to assign an IDENTITY them, is necessary to identify their invariants.

Under a local context, we can state that, "neural networks of the same type and with the same hyperparameters are
the same entity". Under this assumption, we can define a locally unique identifier for each entity, calculated from
it's type and it's hyperparameters, this identifier is called a HASH, and it's the first step to define a global unique
identifier for each entity in a machine learning system.

In order to help with this task, the `torchsystem.registry` module provides a set of functions to register pytorch objects,
so when they are initialized, the arguments that were passed to the constructor are stored as metadata to be used later
to calculate their HASH. This module is provided by the [mlregistry](https://github.com/mr-mapache/ml-registry), and it's
documentation can be found here: [https://mr-mapache.github.io/ml-registry/](https://mr-mapache.github.io/ml-registry/).

Example:
```python
from torch import Tensor
from torch.nn import Module
from torch.nn import Linear, Dropout
from torch.nn import ReLU
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchsystem.registry import register, getarguments, gethash

class MLP(Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float, activation: Module):
        super().__init__()
        self.input_layer = Linear(input_size, hidden_size, bias=True)
        self.dropout = Dropout(dropout)
        self.activation = activation
        self.output_layer = Linear(hidden_size, output_size)
    
    def forward(self, features: Tensor):
        features = self.input_layer(features)
        features = self.dropout(features)
        features = self.activation(features)
        features = self.output_layer(features)
        return features 

register(ReLU)
register(MLP)
register(CrossEntropyLoss)
register(Adam, excluded_args=[0])

model = MLP(784, 256, 10, dropout=0.5, activation=ReLU())
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

print(gethash(model)) # af51a51a38f7ad81f9523360fafe7068
print(getarguments(model)) # {'input_size': 784, 'hidden_size': 256, 'output_size': 10, 'dropout': 0.5, 'activation': 'ReLU'
print(getarguments(criterion)) # {}
print(getarguments(optimizer)) # {'lr': 0.001}
```

### Retrieve your models from the registry.

You can also register classes in a `Registry` object. This will allow you to retrieve the classes by their name. This is
useful for example when you want to load a model from a configuration file, or you want to expose them through a REST API.

Example:
```python

from torchsystem.registry import Registry
from torchsystem.registry import getclass

registry = Registry()
registry.register(MLP)
registry.register(GLU)

model_type = registry.get('MLP')
model = model_type(784, 256, 10, dropout=0.5, activation=ReLU())
avaliables = registry.keys()
print(avaliables) # ['MLP', 'GLU']
for model in avaliables: 
    print(registry.signature(model))

# {'input_size': int, 'hidden_size': int, 'output_size': int, 'dropout': float, 'activation': 'Module'}
# {'input_size': int, 'hidden_size': int, 'output_size': int, 'dropout': float, 'activation': 'Module'}
```