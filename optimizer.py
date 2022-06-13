import itertools
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from network import Network


class Optimizer:

    def __init__(self, data: dict, config: dict, epochs: int):
        self.data: dict = data['data']
        self.input_dim: int = data['input']
        self.output_dim: int = data['output']
        self.config: dict = config
        self.epochs: int = epochs
        self.num_hyperparams: int = len(config)

    def get_minibatch(self, size=1):
        return self.data[:size, :self.input_dim], self.data[:size, self.input_dim:]

    def run_experiment(self, params):
        # this part needs to be adapted to interface with Deneb
        # this method is equivalent to exp.fit
        loss_func = params['loss']()
        optimizer = params['optim']
        lr = params['lr']
        batch_size = params['batch_size']
        activation = params['activation']
        lam = params['dropout'] if 'dropout' in params.keys() else 0
        # reg = params['reg'] if 'reg' in params.keys() else None
        arch = params['architecture']

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
                raise 'Config Error'

        cartesian_product = list(itertools.product(*axes))
        print(f'Grid searching with {len(cartesian_product)} input points')

        losses = []
        best = (float('inf'), None, None)

        for i, tup in enumerate(cartesian_product):
            if i % 25 == 0:
                print('starting trial', i)

            # Configure a new set of params at each point
            params = {}
            for j in range(self.num_hyperparams):
                param = list(self.config.keys())[j]
                choice = tup[j]
                params[param] = choice

            # print(params)
            loss, model = self.run_experiment(params)
            losses.append(math.log(abs(loss.item()) + 1e-7))
            if loss < best[0]:
                best = (loss, model, params)

        plt.figure()
        plt.title('Grid search')
        plt.xlabel('trial #')
        plt.ylabel('ln of loss')
        plt.plot(losses)
        plt.show()

        return best

    def random_search_optimize(self, trials):
        best = (float('inf'), None, None)
        losses = []

        print(f'Random searching through {trials} input points')
        for i in range(trials):

            if i % 25 == 0:
                print('starting trial', i)
            # configure a set of params from random sample and then run experiment
            params = {}

            for param in self.config.keys():
                data = self.config[param]
                if data['type'] == 'categorical':
                    choice = random.sample(data['values'], 1)[0]
                elif data['type'] == 'int':
                    choice = random.randint(data['min'], data['max'])
                elif data['type'] == 'continuous':
                    choice = random.uniform(data['min'], data['max'])
                else:
                    raise 'Config Error'

                params[param] = choice

            loss, model = self.run_experiment(params)
            losses.append(math.log(abs(loss.item()) + 1e-7))
            if loss < best[0]:
                best = (loss, model, params)

        plt.figure()
        plt.title('Random search')
        plt.xlabel('trial #')
        plt.ylabel('ln of loss')
        plt.plot(losses)
        plt.show()

        return best
