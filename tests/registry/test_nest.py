from torchsystem.registry import register
from torchsystem.registry.accessors import getarguments, gethash, getname

@register
class Module:
    def __init__(self, x: int, y: float, z, t: str = '5'):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

    def params(self):
        return self.x, self.y, self.z, self.t

@register
class Model:
    def __init__(self, module: Module):
        self.module = module

@register
class Dummy:
    def __init__(self):
        pass

def test_nested():
    module = Module(1, 2.0, '3', '4')
    model = Model(module)

    assert getarguments(module) == {'x': 1, 'y': 2.0, 'z': '3', 't': '4'}
    assert getarguments(model) == {'module': {
        'name': 'Module',
        'arguments': {'x': 1, 'y': 2.0, 'z': '3', 't': '4'}}
    }

    model = Model(Dummy())
    assert getarguments(model) == {'module': 'Dummy'}