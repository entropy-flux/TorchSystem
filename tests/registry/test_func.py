from torchsystem.registry import register
from torchsystem.registry.accessors import sethash, setname
from torchsystem.registry.accessors import getarguments, gethash, getname, getmetadata

@register
class Model:
    def __init__(self, x: int, y: float, z, t: str = '5'):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

    def params(self):
        return self.x, self.y, self.z, self.t

@register('Criterion')
class Loss:
    def __init__(self, x: int, y: float, z, t: str = '5'):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

class Optimizer:
    def __init__(self, x: int, y: float, z, t: str = '5'):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

def test_registration():
    model = Model(1, 2.0, '3')
    assert getname(model) == 'Model'
    assert getarguments(model) == {'x': 1, 'y': 2.0, 'z': '3'}
    assert gethash(model) == 'b12461be073bff9f5847f3f423767aa2'

    assert model.params() == (1, 2.0, '3', '5')

    loss = Loss(1, 2.0, '3')
    assert getname(loss) == 'Criterion'
    assert getarguments(loss) == {'x': 1, 'y': 2.0, 'z': '3'}
    assert gethash(loss) != 'b12461be073bff9f5847f3f423767aa2'

    register(Optimizer, excluded_args=[0])
    optimizer = Optimizer(1, 2.0, '3')
    assert getname(optimizer) == 'Optimizer'
    assert getarguments(optimizer) == {'y': 2.0, 'z': '3'}

    setname(model)
    sethash(model)
    assert getmetadata(model) == {
        'name': 'Model',
        'hash': 'b12461be073bff9f5847f3f423767aa2',
        'arguments': {'x': 1, 'y': 2.0, 'z': '3'}
    }