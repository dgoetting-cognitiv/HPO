from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

space = [Integer(1, 5, name='max_depth'),
         Real(10 ** -5, 10 ** 0, "log-uniform", name='learning_rate'),
         Integer(1, 20, name='max_features'),
         Integer(2, 100, name='min_samples_split'),
         Integer(1, 100, name='min_samples_leaf'),
         Categorical(['yes', 'no', 'maybe'], name='answers')]


@use_named_args(space)
def objective(**params):
    print(params)
    raise 'done'
