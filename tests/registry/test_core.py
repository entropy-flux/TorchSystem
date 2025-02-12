from torchsystem.registry.core import cls_signature 
from torchsystem.registry.core import cls_override_init

class Model:
    def __init__(self, x: int, y: float, z, t: str = '5'):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

class Model2:
    def __init__(self, x: int, y: float, z, t: str = '5'):
        self.x = x
        self.y = y
        self.z = z
        self.t = t

def test_cls_signature():
    signature = cls_signature(Model, [0], {"t"})
    print(signature)
    assert signature == {"y": "float", "z": "Any"}

    signature = cls_signature(Model, [1])
    assert signature == {"x": "int", "z": "Any", "t": "str"}

    signature = cls_signature(Model, excluded_kwargs={"z"})
    assert signature == {"x": "int", "y": "float", "t": "str"}

def test_object_parameters():
    cls_override_init(Model, excluded_args=[0], excluded_kwargs={"t"})
    model = Model(1, 2.0, "3")
    assert getattr(model, '__model__arguments__') == {"y": 2.0, "z": "3"}

    cls_override_init(Model2, excluded_args=[1], excluded_kwargs={"y"})
    model = Model2(1, 2.0, "3")
    assert getattr(model, '__model__arguments__') == {"x": 1, "z": "3"}