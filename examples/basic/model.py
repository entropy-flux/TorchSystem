from torch import Tensor
from torch.nn import Flatten
from torch.nn import Module
from torch.nn import Linear
from torch.nn import Dropout
from torch.nn import ReLU

class MLP(Module):
    activations = {
        'relu': lambda: ReLU()
    }

    def __init__(self, input_size: int, hidden_size: int, output_size: int, p: float, activation: str):
        super().__init__()
        self.flatten = Flatten()
        self.input_layer = Linear(input_size, hidden_size)
        self.dropout = Dropout(p=p)
        self.activation = self.activations.get(activation)()
        self.hidden_layer = Linear(hidden_size, hidden_size)
        self.output_layer = Linear(hidden_size, output_size)

    def forward(self, features: Tensor):
        features = self.flatten(features)
        features = self.input_layer(features)
        features = self.dropout(features)
        features = self.activation(features)
        features = self.hidden_layer(features)
        features = self.dropout(features)
        features = self.activation(features)
        return self.output_layer(features)