import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as opt
from matplotlib import pyplot as plt
from skopt.space import Integer, Real, Categorical

from network import Architecture
from optimizer import Optimizer

data = {'data': torch.stack((torch.tensor([0, 0, 1], dtype=torch.float),
                             torch.tensor([0, 1, 0], dtype=torch.float),
                             torch.tensor([1, 0, 0], dtype=torch.float),
                             torch.tensor([1, 1, 1], dtype=torch.float),
                             torch.tensor([0.05, 0.05, 0.95], dtype=torch.float),
                             torch.tensor([0.05, 0.95, 0.05], dtype=torch.float),
                             torch.tensor([0.95, 0.05, 0.05], dtype=torch.float),
                             torch.tensor([0.95, 0.95, 0.95], dtype=torch.float),
                             torch.tensor([0.5, 0.5, 0.5], dtype=torch.float)),
                            dim=0),
        'input': 2,
        'output': 1}

example_config = {
    'lr': {'type': 'continuous', 'max': 0.1, 'min': 10e-4, 'steps': 5},
    'optim': {'type': 'categorical', 'values': [opt.Adam, opt.RMSprop]},
    'batch_size': {'type': 'int', 'max': 9, 'min': 1, 'steps': 5},
    'architecture': {'type': 'categorical',
                     'values': [Architecture([10, 8, 6, 4]), Architecture([8, 6, 4]),
                                Architecture([32, 16])]},
    'dropout': {'type': 'continuous', 'max': 0.5, 'min': 0, 'steps': 5},
    'activation': {'type': 'categorical', 'values': [F.hardswish, torch.tanh, torch.sigmoid, F.leaky_relu_]},
    'loss': {'type': 'categorical', 'values': [torch.nn.BCELoss]}
}

space = [Real(10e-4, 0.1, "log-uniform", name='lr'),
         Categorical([opt.Adam, opt.RMSprop], name='optim'),
         Integer(1, 9, name='batch_size'),
         Categorical([Architecture([10, 8, 6, 4]), Architecture([8, 6, 4]), Architecture([12]),
                      Architecture([32, 16]), Architecture([32, 16, 8, 3])], name='architecture'),
         Real(0, 0.75, name='dropout'),
         Categorical([F.hardswish, torch.tanh, torch.sigmoid, torch.relu], name='activation'),
         Categorical([torch.nn.MSELoss], name='loss')]

optimizer = Optimizer(space, data, epochs=50)


def simple_experiment():
    start = time.time()
    (loss, model, params), _, _ = optimizer.bayesian_optimize(trials=100, initial_pts=15)
    print(f'Finished Bayesian search in {time.time() - start} seconds')

    print(f'Optimal params \n {params}')

    x, y = optimizer.get_minibatch(9)
    print(
        f'Final validation: \ninput: {x} \noutput: {model(x)} \nlabels: {y} \nloss: {loss}')


def bayes_vs_random_average():
    NUM_EXPERIMENTS = 5
    NUM_EVALUATIONS = 50

    results_ra = np.ndarray((NUM_EXPERIMENTS, NUM_EVALUATIONS))
    prior_params = []
    for i in range(NUM_EXPERIMENTS):
        (_, _, p), _, bests = optimizer.random_search_optimize(NUM_EVALUATIONS)
        results_ra[i, :] = np.clip(bests, float('-inf'), 0)
        prior_params.append(p)

    plt.figure()
    plt.title("Best loss over time")
    plt.xlabel("trial #")
    plt.ylabel('mean and std of the loss')

    mean_ra = results_ra.mean(axis=0)
    std_ra = results_ra.std(axis=0)
    plt.plot(mean_ra, color='blue', label='Random')
    plt.fill_between(range(NUM_EVALUATIONS), mean_ra + std_ra, mean_ra - std_ra, facecolor='blue', alpha=0.2)

    print(f'random search finished with {mean_ra[-1]} \nparams are {prior_params[0]}')

    # results_bo = np.ndarray((NUM_EXPERIMENTS, NUM_EVALUATIONS))
    # for i in range(NUM_EXPERIMENTS):
    #     (_, _, p), losses, bests = optimizer.bayesian_optimize(NUM_EVALUATIONS, 10, acq_func='PI')
    #     results_bo[i, :] = np.clip(bests, float('-inf'), 0)
    #
    # mean_bo = results_bo.mean(axis=0)
    # std_bo = results_bo.std(axis=0)
    # plt.plot(mean_bo, color='red', label='Bayes')
    # plt.fill_between(range(NUM_EVALUATIONS), mean_bo + std_bo, mean_bo - std_bo, facecolor='red', alpha=0.2)
    # print(f'Bayesian search finished with {mean_bo[-1]} \n params are {prior_params[-1]}')

    results_prior = np.ndarray((NUM_EXPERIMENTS, NUM_EVALUATIONS))
    for i in range(NUM_EXPERIMENTS):
        prior_points = [list(p.values()) for p in prior_params]
        y_vals = [optimizer.run_experiment(p)[0].item() for p in prior_params]

        (_, _, p), losses, bests = optimizer.bayesian_optimize(NUM_EVALUATIONS, 5, x0=prior_points,
                                                               y0=y_vals, acq_func='LCB', kappa=2)
        results_prior[i, :] = np.clip(bests, float('-inf'), 0)

    mean_pr = results_prior.mean(axis=0)
    std_pr = results_prior.std(axis=0)

    plt.plot(mean_pr, color='green', label='Bayes with priors')
    plt.fill_between(range(NUM_EVALUATIONS), mean_pr + std_pr, mean_pr - std_pr, facecolor='green', alpha=0.2)

    print(f'Bayesian search with priors finished with {mean_pr[-1]} \nparams are {p}')
    plt.legend()
    plt.show()


# simple_experiment()
bayes_vs_random_average()
