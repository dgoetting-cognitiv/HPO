from optimizer import Optimizer
import torch
import torch.optim as opt
import torch.nn.functional as F

data = {'data': torch.stack((torch.tensor([0, 0, 1], dtype=torch.float),
                             torch.tensor([0, 1, 0], dtype=torch.float),
                             torch.tensor([1, 0, 0], dtype=torch.float),
                             torch.tensor([1, 1, 1], dtype=torch.float)), dim=0),
        'input': 2,
        'output': 1}

example_config = {
    'lr': {'type': 'continuous', 'max': 0.05, 'min': 0.001},
    'optim': {'type': 'categorical', 'values': [opt.Adam, opt.SGD, opt.RMSprop]},
    'batch_size': {'type': 'int', 'max': 4, 'min': 1, 'step': 8},
    'architecture': {'type': 'categorical', 'values': [[4, 2], [8, 4], [8, 6, 4], [16, 4, 2]]},
    'dropout': {'type': 'continuous', 'max': 0.75, 'min': 0},
    'activation': {'type': 'categorical', 'values': [F.relu, torch.sigmoid, F.hardswish]},
    'loss': {'type': 'categorical', 'values': [torch.nn.BCELoss]}
}

optimizer = Optimizer(data, example_config, 50)
loss, model, params = optimizer.random_search_optimize(100)
print(loss)
print(params)
