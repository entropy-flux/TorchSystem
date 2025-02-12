from torchsystem.registry import Registry

class FooBar:
    x: int
    y: float
    z: str

registry = Registry[FooBar]()

@registry.register
class Foo:
    def __init__(self, x: int, y: float, z: str):
        self.x = x
        self.y = y
        self.z = z

@registry.register('bar', excluded_args=[0], excluded_kwargs=['z'])
class Bar:
    def __init__(self, x: int, y: float, z: str):
        self.x = x
        self.y = y
        self.z = z

def test_registry():
    instance = registry.get('Foo')(1, 2.0, '3') 
    assert instance.x == 1
    assert instance.y == 2.0
    assert instance.z == '3'

    instance = registry.get('bar')(1, 2.0, '3')
    assert instance.x == 1
    assert instance.y == 2.0
    assert instance.z == '3'

    signature = registry.signature('Foo')
    assert signature == {'x': 'int', 'y': 'float', 'z': 'str'}

    signature = registry.signature('bar')
    assert signature == {'y': 'float'}