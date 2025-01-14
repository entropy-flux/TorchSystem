from typing import Optional
from typing import Union
from typing import Iterable
from typing import Literal
from typing import Callable
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.utils.data import Sampler

class WeightsSettings(BaseSettings):
    directory: str = Field('data/weights', description='The path to store the weights.')

class StorageSettings(BaseSettings):
    folder: Optional[str] = Field(default=None)
    weights: WeightsSettings = Field(default_factory=WeightsSettings)

class CompilerSettings(BaseSettings):
    fullgraph: bool = Field(default=False)
    dynamic: Optional[Union[bool, None]] = Field(default=None, description="Use dynamic shape tracing")
    backend: Union[str, Callable] = Field(default='inductor')
    mode: Literal['default', 'reduce-overhead', 'max-autotune','max-autotune-no-cudagraphs'] = Field(default='default')
    options: Optional[dict[str, Union[str, bool]]] = Field(default=None)
    disable: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_prefix='COMPILER_',
        arbitrary_types_allowed=True,
    )


class DatasetsSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_prefix='DATASETS_',
        arbitrary_types_allowed=True,
    )
    
class LoadersSettings[T: DatasetsSettings](BaseSettings):
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
        env_prefix='LOADERS_',
        arbitrary_types_allowed=True,
    )

class AggregateSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_prefix='AGGREGATE_',
        arbitrary_types_allowed=True,
    )

class Settings[T: AggregateSettings](BaseSettings):
    storage: StorageSettings = Field(default_factory=StorageSettings)
    loaders: LoadersSettings = Field(default_factory=LoadersSettings)
    compiler: CompilerSettings = Field(default_factory=CompilerSettings)
    aggregate: Optional[T] = Field(default=None)