from unittest.mock import Mock
from torchsystem.depends import inject, Depends, Provider

openmock = Mock()
closemock = Mock()

def normal_dependency():
    return 42

def generator_dependency():
    openmock()
    yield 42
    closemock()

provider = Provider()

@inject(provider)
def normal_function(dependency = Depends(normal_dependency)):
    return dependency

@inject(provider)
def generator_function(dependency = Depends(generator_dependency)):
    return dependency

def test_normal_dependency():
    assert normal_function() == 42

def test_generator_dependency():
    assert generator_function() == 42
    openmock.assert_called_once()
    closemock.assert_called_once()

overrideopenmock = Mock()
overrideclosemock = Mock()

def override_normal_dependency_with_generator():
    overrideopenmock()
    yield 43
    overrideclosemock()


def test_dependency_override():
    provider.dependency_overrides[normal_dependency] = override_normal_dependency_with_generator
    assert normal_function() == 43
    overrideopenmock.assert_called_once()
    overrideclosemock.assert_called_once() 

def left_leaf_dependency():
    return 2
    
def right_leaf_dependency():
    return 3

def right_node_dependency(leaf = Depends(right_leaf_dependency)):
    return leaf*5

def root_dependency(left = Depends(left_leaf_dependency), right = Depends(right_node_dependency)):
    return left*right*7

@inject(provider)
def handle_dependency(root = Depends(root_dependency)):
    return root*11

def test_handle_nested_depends():
    value = handle_dependency()
    assert value == 2*3*5*7*11

    provider.dependency_overrides[right_node_dependency] = lambda: 13
    value = handle_dependency()
    assert value == 2*7*11*13  