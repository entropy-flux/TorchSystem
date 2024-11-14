from typing import Iterator
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchsystem.aggregate import Loader
from torchsystem.settings import Settings
from mlregistry import Registry

#TODO: Add more infrastructure parameters here and in the settings. All
# DataLoader infrastructure parameters should be configurable.
INFRASTRUCTURE_PARAMETERS = {'dataset', 'pin_memory', 'pin_memory_device' ,'num_workers'}

class Loaders:
    '''
    This class acts as an iterable container for the data loaders. It allows decoupling the infrastructure
    settings like devices from the domain settings of the dataloaders like batch size, shuffling, etc. When
    you add a dataset to the loaders it will automatically register the dataset and it's dataloader in the
    registry and add metadata to them.
    '''

    def __init__(self, settings: Settings = None, exclude_parameters: set[str] = INFRASTRUCTURE_PARAMETERS):
        self.settings = settings or Settings()
        self.registry = Registry(excluded_positions=[0], exclude_parameters=exclude_parameters)
        self.registry.register(DataLoader, 'loader')
        self.list = list[tuple[str, Loader]]()
    
    def add(self, phase: str, dataset: Dataset, batch_size: int, shuffle: bool = False, settings: Settings | None = None, **kwargs):
        '''
        Add a dataset to the loaders. The dataset is added with the phase, batch size, and shuffle settings.

        Parameters:
            phase: str: The phase of the dataset. It can be 'train', 'validation', or 'test'.
            dataset: Dataset: The dataset to be added to the loaders.
            batch_size: int: The batch size of the dataset.
            shuffle: bool: Whether to shuffle the dataset or not.
            settings: Settings: The settings to be used for the dataloader. If None, the default settings are used.
            **kwargs: Any: Additional keyword arguments to be passed to the DataLoader.
        '''
        settings = settings or self.settings
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            pin_memory=self.settings.loaders.pin_memory,
            pin_memory_device=self.settings.loaders.pin_memory_device,
            num_workers=self.settings.loaders.num_of_workers,
            **kwargs
        )
        self.list.append((phase, loader))

    def __iter__(self) -> Iterator[tuple[str, Loader]]:
        return iter(self.list)
    
    def clear(self):
        self.list.clear()