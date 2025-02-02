### TinySys

This is a very basic example of a training service, using TinyDB as storage for the data of the trained models, datasets, dataloaders configuration, metrics, criterions and optimizers used. Also logs metrics using `tensorboard`. The important stuff is in the `tinysys` folder. In the `ports` folders you can find the interfaces of the data model and in the `adapters` their implementation. In the `services`folder you can find:
- The training service
- A compilation pipeline
- A tensorboard consumer
- The data storage consumer.
- A logging service decoupled from the training logic.

Metrics are calculated using the `torcheval` package. Weights are stored using a very simple repository injected into the service. 