from torch import Tensor
from torch.nn import Module
from torch.nn import Linear, Dropout 

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