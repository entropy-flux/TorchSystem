from typing import Callable, Generator
from inspect import signature
from contextlib import ExitStack, contextmanager

class Provider:
    def __init__(self):
        self.dependency_overrides = dict()

class Dependency:
    def __init__(self, callable: Callable):
        self.callable = callable

def resolve(function: Callable, provider: Provider, *args, **kwargs):
    parameters = signature(function).parameters
    bounded = signature(function).bind_partial(*args, **kwargs)
    exit_stack = ExitStack()
    
    for name, parameter in parameters.items():
        if name not in bounded.arguments and isinstance(parameter.default, Dependency):
            dependency = parameter.default.callable
            if dependency in provider.dependency_overrides:
                dependency = provider.dependency_overrides[dependency]
            
            dep_instance = dependency()
            
            if isinstance(dep_instance, Generator):
                bounded.arguments[name] = exit_stack.enter_context(_managed_dependency(dep_instance))
            else:
                bounded.arguments[name] = dep_instance
    
    return bounded, exit_stack

@contextmanager
def _managed_dependency(generator: Generator):
    try:
        value = next(generator)
        yield value
    finally:
        next(generator, None)  # Ensure proper cleanup

def Depends(callable: Callable):
    return Dependency(callable)

def inject(provider: Provider):
    def decorator(function: Callable):
        def wrapper(*args, **kwargs):
            bounded, exit_stack = resolve(function, provider, *args, **kwargs)
            with exit_stack:
                return function(*bounded.args, **bounded.kwargs)
        return wrapper
    return decorator
