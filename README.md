# torch-system

An IA training system, created using domain driven design and an event driven architecture.

## Installation

Make sure you have a pytorch distribution installed. If you don't, go to the [official website](https://pytorch.org/) and follow the instructions.
    
Then, you can install the package using pip:

```bash
pip install torchsystem
```

Soon I will be adding the package to conda-forge when the package is more stable.

## Introduction

Machine learning systems are getting more and more complex, and the need for a more organized and structured way to build and maintain them is becoming more evident. Training a neural network requires to define a cluster of related objects that should be treated as a single unit, this defines an aggregate. The training process mutates the state of the aggregate producing data that should be stored  alongside the state of the aggregate in a transactional way. This establishes a clear bounded context that should be modeled using Domain Driven Design (DDD) principles.

The torch-system is a framework based on DDD and Event Driven Architecture (EDA) principles, using the ![pybondi](https://github.com/mapache-software/py-bondi) library. It aims to provide a way to model complex machine models using aggregates and training flows using commands and events, and persist states and results using the repositories, the unit of work pattern and pub/sub.

It also provides out of the box tools for managing the training process, model compilation, centralized settings with enviroments variables using ![pydantic-settings](https://github.com/pydantic/pydantic-settings), automatic parameter tracking using ![mlregistry](https://github.com/mapache-software/ml-registry). 

