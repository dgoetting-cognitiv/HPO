import time

import torch
import torch.nn.functional as F
import torch.optim as opt
from skopt.space import Integer, Real, Categorical

from network import Architecture
from optimizer import Optimizer

data = {'data': torch.stack((torch.tensor([0, 0, 1], dtype=torch.float),
                             torch.tensor([0, 1, 0], dtype=torch.float),
                             torch.tensor([1, 0, 0], dtype=torch.float),
                             torch.tensor([1, 1, 1], dtype=torch.float)), dim=0),
        'input': 2,
        'output': 1}

example_config = {
    'lr': {'type': 'continuous', 'max': 0.1, 'min': 0.001, 'steps': 5},
    'optim': {'type': 'categorical', 'values': [opt.Adam, opt.RMSprop]},
    'batch_size': {'type': 'int', 'max': 4, 'min': 2, 'steps': 3},
    'architecture': {'type': 'categorical',
                     'values': [Architecture([10, 8, 6, 4]), Architecture([8, 6, 4]), Architecture([16, 4, 2]),
                                Architecture([32, 16])]},
    'dropout': {'type': 'continuous', 'max': 0.75, 'min': 0, 'steps': 5},
    'activation': {'type': 'categorical', 'values': [F.hardswish, torch.tanh]},
    'loss': {'type': 'categorical', 'values': [torch.nn.BCELoss]}
}

space = [Real(10e-5, 10e0, "log-uniform", name='lr'),
         Categorical([opt.Adam, opt.RMSprop], name='optim'),
         Integer(2, 4, name='batch_size'),
         Categorical([Architecture([10, 8, 6, 4]), Architecture([8, 6, 4]), Architecture([16, 4, 2]),
                      Architecture([32, 16])], name='architecture'),
         Real(0, 0.75, name='dropout'),
         Categorical([F.hardswish, torch.tanh], name='activation'),
         Categorical([torch.nn.BCELoss], name='loss')]

optimizer = Optimizer(data, example_config, 50, space)

# start = time.time()
# loss_gr, model_gr, params_gr = optimizer.grid_search_optimize()
# print(f'Finished grid search in {time.time() - start} seconds')
#
# start = time.time()
# loss_ra, model_ra, params_ra = optimizer.random_search_optimize(50)
# print(f'Finished random search in {time.time() - start} seconds')
#

start = time.time()
print(optimizer.bayesian_optimize())
print(f'Finished Bayesian search in {time.time() - start} seconds')
