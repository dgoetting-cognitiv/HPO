import itertools
import math
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skopt import gp_minimize
from skopt.space import Dimension
from skopt.utils import use_named_args

from network import Network, Architecture


class Optimizer:
    expected_params = {'lr', 'optim', 'batch_size', 'architecture', 'activation', 'dropout', 'loss'}

    def __init__(self, data: dict, config: dict, epochs: int, space: List[Dimension]):
        self.best_losses = None
        self.losses = None
        self.best = None
        self.data: torch.Tensor = data['data']
        self.input_dim: int = data['input']
        self.output_dim: int = data['output']
        self.space = space
        if self.input_dim + self.output_dim != self.data.shape[1]:
            raise 'Error, data should include inputs and labels'

        if config.keys() != self.expected_params:
            raise 'Unexpected param configurations'

        self.config: dict = config
        self.epochs: int = epochs
        self.num_hyperparams: int = len(config)
        self.count = 0

    def get_minibatch(self, size=1):
        return self.data[:size, :self.input_dim], self.data[:size, self.input_dim:]

    def run_experiment(self, params):
        # this part needs to be adapted to interface with Deneb
        # this method is equivalent to exp.fit
        assert params.keys() == self.expected_params
        self.count += 1
        if self.count % 5 == 0:
            print(f'Running trial number {self.count}')

        lr: float = params['lr']
        optimizer = params['optim']
        batch_size: int = params['batch_size']
        arch: [Architecture] = params['architecture']
        activation = params['activation']
        lam: float = params['dropout'] if 'dropout' in params.keys() else 0
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
        x, y = self.get_minibatch(4)
        pred = net(x)
        validate_loss = nn.BCELoss()
        loss = validate_loss(pred, y)

        return loss, net

    def grid_search_optimize(self):

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

        losses = []
        best = (float('inf'), None, None)
        best_losses = []

        for i, tup in enumerate(cartesian_product):
            if i % 25 == 0:
                print(f'starting trial {i}, best loss so far is {best[0]}')

            # Configure a new set of params at each point
            params = {}
            for j in range(self.num_hyperparams):
                param = list(self.config.keys())[j]
                choice = tup[j]
                params[param] = choice

            loss, model = self.run_experiment(params)
            losses.append(math.log(abs(loss.item()) + 1e-7))
            if loss < best[0]:
                best = (loss, model, params)
            best_losses.append(math.log(abs(best[0].item()) + 1e-7))

        plt.figure()
        plt.title('Grid search')
        plt.xlabel('trial #')
        plt.ylabel('ln of loss')
        plt.plot(losses)
        plt.plot(best_losses)
        plt.savefig('grid.png')
        plt.show()

        return best

    def random_search_optimize(self, trials):
        best = (float('inf'), None, None)
        losses = []
        best_losses = []

        print(f'Random searching through {trials} input points')
        for i in range(trials):

            if i % 25 == 0:
                print(f'starting trial {i}, best loss so far is {best[0]}')
            # configure a set of params from random sample and then run experiment
            params = self.sample_random()

            loss, model = self.run_experiment(params)
            losses.append(math.log(abs(loss.item()) + 1e-7))
            if loss < best[0]:
                best = (loss, model, params)
            best_losses.append(math.log(abs(best[0].item()) + 1e-7))

        plt.figure()
        plt.title('Random search')
        plt.xlabel('trial #')
        plt.ylabel('ln of loss')
        plt.plot(losses)
        plt.plot(best_losses)
        plt.savefig('rand.png')
        plt.show()

        return best

    def sample_random(self):
        params = {}
        params2vec = []

        for param in self.config.keys():
            data = self.config[param]
            if data['type'] == 'categorical':
                ndx = random.randint(0, len(data['values']) - 1)
                choice = data['values'][ndx]
                params2vec.append(ndx)
            elif data['type'] == 'int':
                choice = random.randint(data['min'], data['max'])
                params2vec.append(choice)
            elif data['type'] == 'continuous':
                choice = random.uniform(data['min'], data['max'])
                params2vec.append(choice)
            else:
                raise 'All params must have type of categorical, int or continuous'

            params[param] = choice

        return params, params2vec

    def bayesian_optimize(self):
        self.count = 0
        self.best = (float('inf'), None, None)
        self.losses = []
        self.best_losses = []

        @use_named_args(self.space)
        def objective(**params):
            loss, model = self.run_experiment(params)
            self.losses.append(math.log(abs(loss.item()) + 1e-7))
            if loss < self.best[0]:
                self.best = (loss, model, params)
            self.best_losses.append(math.log(abs(self.best[0].item()) + 1e-7))
            return loss.item()

        gp_minimize(objective, self.space, n_calls=50, n_initial_points=20)

        plt.figure()
        plt.title('Bayesian search')
        plt.xlabel('trial #')
        plt.ylabel('ln of loss')
        plt.plot(self.losses)
        plt.plot(self.best_losses)
        plt.savefig('rand.png')
        plt.show()

        return self.best
