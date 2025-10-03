from torch import Tensor
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Linear, Dropout 
from torch.nn import Flatten

class MLP(Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float):
        super().__init__()
        self.input_layer = Linear(input_size, hidden_size, bias=True)
        self.dropout = Dropout(dropout)
        self.activation = ReLU()
        self.output_layer = Linear(hidden_size, output_size)
        self.flatten = Flatten()
    
    def forward(self, features: Tensor):
        features = self.flatten(features)
        features = self.input_layer(features)
        features = self.dropout(features)
        features = self.activation(features)
        features = self.output_layer(features)
        return features