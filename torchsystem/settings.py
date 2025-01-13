from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class WeightsSettings(BaseSettings):
    directory: str = Field('data/weights', description='The path to store the weights.')

class CompilerSettings(BaseSettings):
    ...

class Settings[T: BaseSettings]:
    compiler: CompilerSettings = Field(default_factory=CompilerSettings)
    aggregate: Optional[T] = Field(default=None)
    weights: WeightsSettings = Field(default_factory=WeightsSettings)