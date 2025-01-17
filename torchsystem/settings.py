"""
Settings module contains the settings for the torchsystem. It centralizes all the settings for
the storage, loaders, compiler and aggregate. The settings are optional and can be overridden by the user.

All settings support environment variables and configuration files thanks to the `pydantic-settings` package.
They also validate fields and provide the same default values as PyTorch's defaults.

Example:
    An example of an `.env` file:

        # Storage settings
        STORAGE_WEIGHTS_DIRECTORY=data/weights

        # Compiler settings
        COMPILER_FULLGRAPH=False
        COMPILER_DYNAMIC=False

        # Loader settings
        LOADERS_NUM_WORKERS=4
        LOADERS_PIN_MEMORY=True
        LOADERS_PIN_MEMORY_DEVICE=cuda

        # Aggregate settings
        AGGREGATE_DEVICE=cuda


Usage:
    To use a `.env` file, create a file named `.env` in your project's root directory with 
    the desired settings. The settings module will automatically load these values at runtime.
"""


from typing import Optional
from typing import Union
from typing import Iterable
from typing import Literal
from typing import Callable
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.utils.data import Sampler

class WeightsSettings(BaseSettings):
    """
    WeightsSettings is a class that contains the settings for the weights.

    Args:
        directory (str): The path to store the weights. Default is 'data/weights'.
    """
    directory: str = Field('data/weights', description='The path to store the weights.')
    
    model_config = SettingsConfigDict(
        env_prefix='weights_',
        arbitrary_types_allowed=True,
        case_sensitive=False
    )


class StorageSettings(BaseSettings):
    """
    StorageSettings is a class that contains the settings for the storage of torch modules.
    It contains the folder for general storage and the weights settings.
    Args:
        folder (Optional[str]): The folder to store the torch modules. Default is None.
        weights (WeightsSettings): The settings for the weights.
    """
    folder: Optional[str] = Field(default=None)
    weights: WeightsSettings = Field(default_factory=WeightsSettings)


    model_config = SettingsConfigDict(
        env_prefix='storage_',
        arbitrary_types_allowed=True,
        case_sensitive=False
    )

class CompilerSettings(BaseSettings):
    """
    CompilerSettings is a class that contains the settings for the `Compiler`. It defaults
    to the pytorch's defaults for the compile function. 

    Args:
        BaseSettings (_type_): _description_
    """
    fullgraph: bool = Field(default=False)
    dynamic: Optional[Union[bool, None]] = Field(default=None, description="Use dynamic shape tracing")
    backend: Union[str, Callable] = Field(default='inductor')
    mode: Literal['default', 'reduce-overhead', 'max-autotune','max-autotune-no-cudagraphs'] = Field(default='default')
    options: Optional[dict[str, Union[str, bool]]] = Field(default=None)
    disable: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_prefix='compiler_',
        arbitrary_types_allowed=True,
        case_sensitive=False
    )


class DatasetsSettings(BaseSettings):
    """
    A base class for the settings of the datasets. It empty and should be inherited by the user for their own settings.
    """

    model_config = SettingsConfigDict(
        env_prefix='datasets_',
        arbitrary_types_allowed=True,
        case_sensitive=False
    )
    
class LoadersSettings[T: DatasetsSettings](BaseSettings):
    """
    LoadersSettings is a class that contains the settings for the dataloaders. It contains
    the same attributes as the `torch.utils.data.DataLoader` class and defaults to the PyTorch's 
    defaults. It also carries the settings for the dataset defined by the user.
    """
    dataset: Optional[T] = Field(default=None)
    sampler: Union[Sampler, Iterable, None] = Field(default=None)
    batch_sampler: Union[Sampler[list], Iterable[list], None] = Field(default=None)
    num_workers: int = Field(default=0)
    collate_fn: Optional[Callable] = Field(default=None)
    pin_memory: bool = Field(default=False)
    timeout: float = Field(default=0.0)
    worker_init_fn: Optional[Callable] = Field(default=None)
    prefetch_factor: Optional[int] = Field(default=None)
    persistent_workers: bool = Field(default=False)  
    pin_memory_device: str = Field(default='')
    
    model_config = SettingsConfigDict(
        env_prefix='loaders_',
        arbitrary_types_allowed=True,
        case_sensitive=False
    )


class AggregateSettings(BaseSettings):
    """
    AggregateSettings is a class that contains the settings for the `Aggregate`. It's empty
    and should be inherited by the user for their own settings.

    Example:
    
        .. code-block:: python
        from torchsystem import AggregateSettings

        class ClassifierSettings(AggregateSettings):
            device: str = 'cpu'

        settings = Settings(aggregate=ClassifierSettings())
    """

    model_config = SettingsConfigDict(
        env_prefix='aggregate_',
        arbitrary_types_allowed=True,
        case_sensitive=False
    )


class Settings[T: AggregateSettings](BaseSettings):
    """
    Central class that contains all the settings for the torchsystem.
    It contains the settings for the storage, loaders, compiler and aggregate.
    All the settings are optional and can be overridden by the user. They support
    environment variables and configuration files. 

    Example:

        .. code-block:: python
        from dotenv import load_dotenv
        from torchsystem import Settings

        load_dotenv()
        settings = Settings() # Will load the settings from the environment variables
    """
    storage: StorageSettings = Field(default_factory=StorageSettings)
    loaders: LoadersSettings = Field(default_factory=LoadersSettings)
    compiler: CompilerSettings = Field(default_factory=CompilerSettings)
    aggregate: Optional[T] = Field(default=None)