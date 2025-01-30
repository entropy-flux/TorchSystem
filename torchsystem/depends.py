#TODO: Handle the following cases:
# 1. The function is a Generator
# 2. The function is a Annotated
# 3. The function is a Coroutine
# 4. The function is a AsyncGenerator

#TODO: Enhace this module redability and maintainability

#TODO: Write documentation for this module

#TODO: Check if this should be adapted to the torch ecosystem
# In order to avoid memory leaks and other issues.
# Especially during distributed training.

from typing import Callable
from inspect import signature

class Provider:
    def __init__(self):
        self.dependency_overrides = dict()

class Dependency:
    def __init__(self, callable: Callable):
        self.callable = callable

def resolve(function: Callable, provider: Provider, *args, **kwargs):
    parameters = signature(function).parameters
    bounded = signature(function).bind_partial(*args, **kwargs)
    
    for name, parameter in parameters.items():
        if name not in bounded.arguments and isinstance(parameter.default, Dependency):
            dependency = parameter.default.callable
            if dependency in provider.dependency_overrides:
                bounded.arguments[name] = provider.dependency_overrides[dependency]()
            else:
                bounded.arguments[name] = dependency()
    return bounded

def Depends(callable: Callable):
    return Dependency(callable)

def inject(provider: Provider):
    def decorator(function: Callable):
        def wrapper(*args, **kwargs):
            bounded = resolve(function, provider, *args, **kwargs)
            return function(*bounded.args, **bounded.kwargs)
        return wrapper
    return decorator