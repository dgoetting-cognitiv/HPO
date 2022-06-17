import itertools
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from skopt import gp_minimize
from skopt.space import Dimension, Categorical
from skopt.utils import use_named_args

from network import Network, Architecture


class Optimizer:
    expected_params = {'lr', 'optim', 'batch_size', 'architecture', 'activation', 'dropout', 'loss'}

    def __init__(self, space: List[Dimension], data: dict, epochs: int = 50, config: dict = None):
        self.best_result = None
        self.best_losses = None
        self.losses = None
        self.data: torch.Tensor = data['data']
        self.input_dim: int = data['input']
        self.output_dim: int = data['output']
        self.space = space
        if self.input_dim + self.output_dim != self.data.shape[1]:
            raise 'Error, data should include inputs and labels'

        if {dim.name for dim in self.space} != self.expected_params:
            raise 'Unexpected param configurations'

        self.config: dict = config
        self.epochs: int = epochs
        self.num_hyperparams: int = len(space)
        self.count = 0

    def get_minibatch(self, size=1):
        shuffled = self.data[torch.randperm(self.data.shape[0])]
        return shuffled[:size, :self.input_dim], shuffled[:size, self.input_dim:]

    def run_experiment(self, params: dict):
        # this part needs to be adapted to interface with Deneb
        # this method is equivalent to exp.fit
        assert params.keys() == self.expected_params
        self.count += 1
        if self.count % 10 == 0:
            print(f'Running trial number {self.count}')

        lr: float = params['lr']
        optimizer = params['optim']
        batch_size: int = params['batch_size']
        arch: [Architecture] = params['architecture']
        activation = params['activation']
        lam: float = params['dropout']
        loss_func = params['loss']()

        net = Network(arch=arch, activation=activation, dropout=nn.Dropout(p=lam))
        optim = optimizer(lr=lr, params=net.parameters())

        for _ in range(self.epochs):
            x, y = self.get_minibatch(batch_size)
            pred = net(x)
            loss = loss_func(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Evaluate
        x, y = self.get_minibatch(self.data.shape[0])
        pred = net(x)
        validate_loss = nn.MSELoss()
        loss = validate_loss(pred, y)

        return loss, net

    def grid_search_optimize(self):
        self.reset_metrics()
        axes = []
        for param in self.config.keys():
            data = self.config[param]
            if data['type'] == 'categorical':
                axes.append(data['values'])
            elif data['type'] == 'int':
                axis = np.linspace(start=data['min'], stop=data['max'], num=data['steps'], dtype=int)
                axes.append(axis)
            elif data['type'] == 'continuous':
                axis = np.linspace(start=data['min'], stop=data['max'], num=data['steps'])
                axes.append(axis)
            else:
                raise 'All params must have type of categorical, int or continuous'

        cartesian_product = list(itertools.product(*axes))
        print(f'Grid searching with {len(cartesian_product)} input points')

        for tup in cartesian_product:
            # Configure a new set of params at each point
            params = self.vec2param(tup)

            loss, model = self.run_experiment(params)
            if loss < self.best_result[0]:
                self.best_result = (loss, model, params)

            self.losses.append(math.log(loss.item() + 1e-15, 2))
            self.best_losses.append(math.log(self.best_result[0].item() + 1e-15, 2))

        return self.best_result, self.losses, self.best_losses

    def random_search_optimize(self, trials):

        self.reset_metrics()

        print(f'Random searching through {trials} input points')
        for i in range(trials):
            # configure a set of params from random sample and then run experiment
            params = self.sample_random()

            loss, model = self.run_experiment(params)
            self.losses.append(math.log(loss.item() + 1e-15, 2))
            if loss < self.best_result[0]:
                self.best_result = (loss, model, params)
                print('found an improvement at time ', self.count)
            self.best_losses.append(math.log(self.best_result[0].item() + 1e-15, 2))

        return self.best_result, self.losses, self.best_losses

    def bayesian_optimize(self, trials, initial_pts, acq_func='PI', acq_optimizer='sampling', n_points=1000, xi=0.005,
                          kappa=1.5, initial_point_generator='halton', x0=None, y0=None):
        self.reset_metrics()
        print(f'Running bayesian search through {trials} points {"WITH PRIORS" if x0 else ""}')

        @use_named_args(self.space)
        def objective(**params):
            loss, model = self.run_experiment(params)
            self.losses.append(math.log(loss.item() + 1e-15, 2))
            if loss < self.best_result[0]:
                self.best_result = (loss, model, params)
                print('found an improvement at time ', self.count)
            self.best_losses.append(math.log(self.best_result[0].item() + 1e-15, 2))
            return loss.item()

        gp_minimize(objective, self.space, n_calls=trials,
                    n_initial_points=initial_pts, acq_func=acq_func, acq_optimizer=acq_optimizer, n_points=n_points,
                    xi=xi, kappa=kappa, initial_point_generator=initial_point_generator, x0=x0, y0=y0)

        return self.best_result, self.losses, self.best_losses

    def reset_metrics(self):
        self.best_losses = []
        self.best_result = (float('inf'), None, None)
        self.losses = []
        self.count = 0

    def sample_random(self):
        params = {}

        for dim in self.space:
            params[dim.name] = dim.rvs(1)[0]

        return params

    def param2vec(self, param):
        params2vec = []
        assert param.keys() == {dim.name for dim in self.space}

        for dim in self.space:
            if isinstance(dim, Categorical):
                params2vec.append(dim.categories.index(param[dim.name]))
            else:
                params2vec.append(param[dim.name])

        return params2vec

    def vec2param(self, vec):
        params = {}

        for i, dim in enumerate(self.space):
            if isinstance(dim, Categorical):
                params[dim.name] = dim.categories[vec[i]]
            else:
                params[dim.name] = vec[i]

        return params
