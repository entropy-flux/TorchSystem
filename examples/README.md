## Examples

### verybasic

Contains a few very basic examples of isolated components of the library.

### tinysys

A basic training system using [tinydb](https://github.com/msiemens/tinydb) for storing loss and accuracy metrics, the datasets, their parameters and loaders configuration of each iteration (creating a new one if some parameter change else updating the epoch), the modules used for the `Classifier` class aside from the neural network with their respective epoch, and models under an experiment namespace. It also produce logs in tensorboard.

If the system finds that you are training a model that is already in the database, it will restore the epochs and the weights. There is a simple main file without any epoch loop logic for showing you the re-storage mechanism. Run it once, and the model will get to the first epoch, run it again with the same nn, and it will start from the first epoch and get to the second, if you change some hyperparameter of the nn, the system will recognize it and start from epoch 0, having storing all data from the prior nn.

If you set the hyperparameters of the nn to a prior trained configuration, the system will be able to fully restore if and start from where it was left off.