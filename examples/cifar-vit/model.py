import math
from typing import Optional
import torch  
from torch import concat
from torch import nn
from torch import Tensor   
from torch import polar, ones_like, empty
from torch import view_as_complex, view_as_real 
from torch.nn.functional import silu 
from torch.nn.functional import scaled_dot_product_attention 

def split(sequence: Tensor, number_of_heads: int) -> Tensor:
    batch_size, sequence_length, model_dimension = sequence.shape
    sequence = sequence.view(batch_size, sequence_length, number_of_heads, model_dimension // number_of_heads)
    return sequence.transpose(1, 2) 

def merge(sequence: Tensor) -> Tensor:
    batch_size, number_of_heads, sequence_length, heads_dimension = sequence.shape
    sequence = sequence.transpose(1, 2)
    return sequence.reshape(batch_size, sequence_length, heads_dimension* number_of_heads)

def apply_frequencies(sequence: Tensor, frequencies: Tensor) -> Tensor:
    batch_size, number_of_heads, sequence_length, heads_dimension = sequence.shape 
    sequence = sequence.view(batch_size, number_of_heads, sequence_length, heads_dimension // 2, 2)
    sequence = view_as_complex(sequence) 
    sequence = sequence * frequencies 
    sequence = view_as_real(sequence) 
    return sequence.view(batch_size, number_of_heads, sequence_length, heads_dimension)  


class Attention(nn.Module): 
    def __init__(self, model_dimension: int, number_of_heads: int):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f"Last dimension ({model_dimension}) must be divisible by number of heads ({number_of_heads})"
        self.q_projection = nn.Linear(model_dimension, model_dimension)
        self.k_projection = nn.Linear(model_dimension, model_dimension)
        self.v_projection = nn.Linear(model_dimension, model_dimension)
        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.heads_dimension = model_dimension // number_of_heads
 
    def forward(self, sequence: Tensor, frequencies: Tensor, mask: Optional[Tensor]): 
        query, key, value = self.q_projection(sequence), self.k_projection(sequence), self.v_projection(sequence)
        query, key, value = split(query, self.number_of_heads), split(key, self.number_of_heads), split(value, self.number_of_heads)
        query, key = apply_frequencies(query, frequencies), apply_frequencies(key, frequencies) 
        output = scaled_dot_product_attention(query, key, value, mask) 
        return merge(output) 

    def split(self, sequence: Tensor) -> Tensor: 
        sequence = sequence.view(sequence.size(0), sequence.size(1), self.number_of_heads, self.model_dimension // self.number_of_heads)
        return sequence.transpose(1, 2)

    def merge(self, sequence: Tensor) -> Tensor:  
        sequence = sequence.transpose(1, 2) 
        return sequence.reshape(sequence.size(0) , sequence.size(1), self.heads_dimension* self.number_of_heads) 


class FFN(nn.Module): 
    def __init__(self, model_dimension: int, hidden_dimension: int):
        super().__init__()  
        self.input_layer   = nn.Linear(model_dimension,  hidden_dimension, bias=False)
        self.gate_layer    = nn.Linear(model_dimension, hidden_dimension, bias=False) 
        self.output_layer  = nn.Linear(hidden_dimension, model_dimension, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        features = silu(self.input_layer(features)) * self.gate_layer(features)
        return self.output_layer(features)


class Encoder(nn.Module): 
    def __init__(self, model_dimension: int, number_of_heads: int, ffn_hidden_dimension: int):
        super().__init__()
        self.attention = Attention(model_dimension, number_of_heads)
        self.attention_norm = nn.LayerNorm(model_dimension, eps=1e-6)
        self.projection = nn.Linear(model_dimension, model_dimension) 
        self.ffn = FFN(model_dimension, ffn_hidden_dimension)
        self.ffn_norm = nn.LayerNorm(model_dimension, eps=1e-6)

    def forward(self, sequence: Tensor, frequencies: Tensor, mask: Optional[Tensor]):    
        sequence = self.projection(self.attention(self.attention_norm(sequence), frequencies, mask)) + sequence 
        return self.ffn(self.ffn_norm(sequence)) + sequence 
 

class Transformer(nn.Module): 
    def __init__(self, number_of_layers: int, model_dimension: int, number_of_heads: int, ffn_hidden_dimension: int):
        super().__init__()
        self.encoders = nn.ModuleList([
            Encoder(model_dimension, number_of_heads, ffn_hidden_dimension) for _ in range(number_of_layers)
        ])

    def forward(self, sequence: Tensor, frequencies: Tensor, mask: Tensor = None):
        for encoder in self.encoders:
            sequence = encoder(sequence, frequencies, mask)
        return sequence  
  
  
class Patcher(nn.Module): 
    def __init__(
        self,
        image_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        input_channels: int = 3,
        model_dimension: int = 768,
    ):
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        ih, iw = self.image_size
        ph, pw = self.patch_size
        assert ih % ph == 0 and iw % pw == 0, "Image dimensions must be divisible by patch size" 
        self.number_of_patches = (ih // ph) *  (iw // pw) 
        self.projection = nn.Conv2d(input_channels, model_dimension, kernel_size=(ph, pw), stride=(ph, pw))

    def forward(self, features: Tensor) -> Tensor:  
        features = self.projection(features)          
        return features.flatten(2).transpose(1, 2)   


class CLSToken(nn.Module): 
    def __init__(self, model_dimension: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, model_dimension)) 

    def forward(self, sequence: Tensor) -> Tensor: 
        batch_size = sequence.size(0)
        cls = self.weight.expand(batch_size, -1, -1)  
        return concat((cls, sequence), dim=1) 


class PositionalEmbedding(nn.Module):  
    def __init__(self, sequence_length: int, model_dimension: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, sequence_length, model_dimension))
    
    def forward(self, sequence: Tensor): 
        return sequence + self.weight
    

def precompute_frequencies(model_dimension: int, sequence_length_limit: int, theta: float = 1000.0):  
    scale = math.log(theta) / model_dimension
    exponents = empty((sequence_length_limit, model_dimension // 2), dtype=torch.float32) 
    for position in range(sequence_length_limit):
        for dimension in range(model_dimension // 2): 
            exponents[position, dimension] = position *  math.exp(-2 * dimension * scale)
    return polar(ones_like(exponents), exponents)


class ViT(nn.Module):  
    frequencies: Tensor

    def __init__(
        self,  
        patch_size:  int,
        input_channels: int, 
        image_size: int,
        model_dimension: int,
        ffn_hidden_dimension: int,
        number_of_heads: int,
        number_of_layers: int,   
        number_of_classes: int
    ):
        super().__init__()
        self.image_size = image_size                 
        self.patcher = Patcher(image_size, patch_size, input_channels, model_dimension) 
        self.cls_token = CLSToken(model_dimension)  
        self.positions =  nn.Parameter(torch.zeros(1, self.patcher.number_of_patches + 1, model_dimension))
        self.transformer = Transformer(number_of_layers, model_dimension, number_of_heads, ffn_hidden_dimension)
        self.norm = nn.LayerNorm(model_dimension, eps=1e-6)
        self.head = nn.Linear(model_dimension, number_of_classes)  
        self.register_buffer("frequencies",
            precompute_frequencies(
                model_dimension // number_of_heads,
                128,
                10000
            ), persistent=False
        )
          
    def forward(self, tokens: Tensor):  
        sequence = self.patcher(tokens)    
        sequence = self.cls_token(sequence) + self.positions   
        sequence = self.transformer(sequence, self.frequencies[0:sequence.size(1)])      
        sequence = self.norm(sequence)    
        return self.head(sequence[:, 0])   