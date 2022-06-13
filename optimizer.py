import math

import torch.nn as nn
import matplotlib.pyplot as plt
from network import Network
import random


class Optimizer:

    def __init__(self, data, config, epochs):
        self.data: dict = data['data']
        self.input_dim: int = data['input']
        self.output_dim: int = data['output']
        self.config: dict = config
        self.epochs: int = epochs

    def get_minibatch(self, size=1):
        return self.data[:size, :self.input_dim], self.data[:size, self.input_dim:]

    def run_experiment(self, params):
        # this part needs to be adapted to interface with Deneb
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
        pass

    def random_search_optimize(self, trials):
        best = (float('inf'), None, None)
        losses = []

        for i in range(trials):

            if i % 10 == 0:
                print('starting trial', i)
            # configure a set of params from random sample and then run experiment
            params = {}

            for param in self.config.keys():
                if self.config[param]['type'] == 'categorical':
                    choice = random.sample(self.config[param]['values'], 1)[0]
                elif self.config[param]['type'] == 'int':
                    choice = random.randint(self.config[param]['min'], self.config[param]['max'])
                elif self.config[param]['type'] == 'continuous':
                    choice = random.uniform(self.config[param]['min'], self.config[param]['max'])
                else:
                    raise 'Config Error'

                params[param] = choice

            loss, model = self.run_experiment(params)
            losses.append(math.log(loss.item()))
            if loss < best[0]:
                best = (loss, model, params)

        plt.figure()
        plt.title('Random search')
        plt.xlabel('trial #')
        plt.ylabel('ln of loss')
        plt.plot(losses)
        plt.show()

        return best
