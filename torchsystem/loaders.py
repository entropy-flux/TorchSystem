from typing import Iterator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchsystem.aggregate import Loader
from mlregistry import Registry

INFRASTRUCTURE_PARAMETERS = {'dataset', 'pin_memory', 'pin_memory_device' ,'num_workers'}

class Loaders:
    def __init__(self, exclude_parameters: set[str] = INFRASTRUCTURE_PARAMETERS):
        self.registry = Registry(excluded_positions=[0], exclude_parameters=exclude_parameters)
        self.registry.register(DataLoader, 'loader')
        self.list = list[tuple[str, Loader]]()
    
    def add(self, phase: str, dataset: Dataset, batch_size: int, shuffle: bool = False, **kwargs):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.list.append((phase, loader))

    def __iter__(self) -> Iterator[tuple[str, Loader]]:
        return iter(self.list)
    
    def clear(self):
        self.list.clear()