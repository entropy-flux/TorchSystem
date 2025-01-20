from torchsystem.compiler import Depends
from torchsystem.compiler import Compiler

compiler = Compiler()

def four():
    raise NotImplementedError

@compiler.step
def add_one_of_tree(a,b,c,d=Depends(four)):
    return a, b + c, d

@compiler.step
def square(a, b, d):
    return a * b * d

@compiler.step
def cuadruple(a, x=Depends(four)):
    return a * x

def test_compiler():
    compiler.dependency_overrides[four] = lambda: 4
    assert compiler.compile(1, 2, 3) == 80