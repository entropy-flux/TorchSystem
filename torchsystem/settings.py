from typing import Callable
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AggregateSettings(BaseSettings):
    device: str = Field(default='cpu')
    model_config = SettingsConfigDict(env_prefix='MODEL_')

class CompilerSettings(BaseSettings):
    raise_on_error: bool = Field(default=False)
    fullgraph: bool = Field(default=False)
    dynamic: bool = Field(default=False)
    backend: str | Callable = Field(default='inductor')
    mode: str | None = Field(default=None)
    options: dict | None = Field(default=None)
    disable: bool = Field(default=False)
    model_config = SettingsConfigDict(env_prefix='COMPILATION_')

class LoaderSettings(BaseSettings):
    pin_memory: bool = Field(default=False)
    pin_memory_device: str = Field(default='')
    num_of_workers: int = Field(default=0)
    model_config = SettingsConfigDict(env_prefix='LOADER_')

class WeightsSettings(BaseSettings):
    directory: str = Field(default='data/weights')
    model_config = SettingsConfigDict(env_prefix='WEIGHTS_')

class Settings[T: BaseSettings](BaseSettings):
    aggregate: T = Field(default_factory=AggregateSettings)
    loaders: LoaderSettings = Field(default_factory=LoaderSettings)
    compilation: CompilerSettings = Field(default_factory=CompilerSettings)
    weights: WeightsSettings = Field(default_factory=WeightsSettings)