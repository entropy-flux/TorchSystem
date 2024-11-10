from mlregistry import Registry

class Base:
    registry: Registry
    category: str
    
    @classmethod
    def register(cls, type: type):
        return cls.registry.register(type, cls.category)

class Models(Base):
    category = 'model'
    registry = Registry()

class Criterions(Base):
    category = 'criterion'
    registry = Registry()

class Optimizers(Base):
    category = 'optimizer'
    registry = Registry(excluded_positions=[0], exclude_parameters={'params'})

class Datasets(Base):
    category = 'dataset'
    registry = Registry(exclude_parameters={'root', 'download'})