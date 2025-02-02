### TinySys

This is a very basic example of a training service, using [tinydb](https://github.com/msiemens/tinydb) as storage for the data of the trained models. tinydb is a simple python database that doesn't require any server, so it's good for a first examples. 

Data from datasets, dataloaders configuration, metrics, criterions and optimizers is stored in the database. Metrics are also logged using `tensorboard`. 

The important stuff is in the `tinysys` folder. In the `ports` folders you can find the interfaces of the data model and in the `adapters` their implementation. In the `services`folder you can find:

- The training service
- A compilation pipeline
- A tensorboard consumer
- The data storage consumer.
- A logging service decoupled from the training logic.

Metrics are calculated using the `torcheval` package. Weights are stored using a very simple repository injected into the storage service. They are restored from the compiler when building the model if the system founds that the same model was being trained under the same experiment namespace.