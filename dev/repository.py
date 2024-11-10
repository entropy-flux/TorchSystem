from torchsystem.storage import Registry
from torchsystem.storage import Storage

class Repository:
    models = Storage('models', registry=Registry())
    criterions = Storage('criterions', registry=Registry())
    optimizers = Storage('optimizers', registry=Registry(excluded_positions=[0], exclude_parameters={'params'}))
    datasets = Storage('datasets', registry=Registry(exclude_parameters={'root', 'download'}))