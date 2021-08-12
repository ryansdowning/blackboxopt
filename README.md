# Black Box Optimization

![Travis-CI](https://img.shields.io/travis/com/ryansdowning/blackboxopt/main)
![Pytest Coverage](https://img.shields.io/codecov/c/github/ryansdowning/blackboxopt)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/ryansdowning/blackboxopt.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/ryansdowning/blackboxopt/context:python)

A package for applying hyperparameter tuning algorithms on a black box function which defines an optimization problem

My attempt at making an all-in-one hyperparameter optimization package. High-level abstraction allows algorithms to be implemented independently of search space constraints. Inspiration pulled from `ray[tune]` and `hyperopt`.

![xkcd-standards](https://imgs.xkcd.com/comics/standards.png)

## Problem

There are currently a handful of Python hyperparameter optimization packages fighting for popularity as popularity rises
with Machine Learning in Python. Many of these packages do something good, some of them even do a lot well, but none of
them are *perfect*. Below is a chart of features that I was looking for when searching for a hyperparameter optimization
framework, as well as a list of packages and how they go about satisfying those features.

|Library|Abstract Function Input|Easy Search Space API|Conditional Search Spaces|Multiple Algorithms|
|-------|:----------------:|:---------:|:---------:|:---------:|
|[BlackBoxOpt](https://ryansdowning.github.io/blackboxopt/)|✅|✅|✅|✅|
|[Ray[Tune]](https://docs.ray.io/en/master/tune/index.html)|✅|✅|✅|⚠️<sup>[1]</sup>|
|[Optuna](https://optuna.org/)|✅|✅|❌|⚠️<sup>[2]</sup>|
|[Hyperopt](https://github.com/hyperopt/hyperopt)|✅|⚠️<sup>[3]</sup>|⚠️<sup>[4]</sup>|⚠️<sup>[5]</sup>|
|[Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)|✅|❌|❌|❌|
|[Scikit-Optimize](https://scikit-optimize.github.io/stable/)|✅|❌|❌|✅|
|[Facebook Ax](https://github.com/facebook/Ax) <sup>[6]</sup>|✅|❌|❌|❌|
[1] - Ray[Tune] implements many optimization algorithms, but fails to support them properly so when you introduce 
conditional search spaces you may only use random sampling.

[2] - Optuna currently supports TPE and CMA-ES sampling algorithms as well as some pruning algorithms. The "Multiple
Algorithms" tag is focused on wider generalization supporting many algorithms (>5).

[3] - Hyperopt attempts to implement conditional spaces in the search space structure itself, making it much more
difficult to use.

[4] - Hyperopt's attempt at implementing conditional search spaces works...but not really how you want it to most of the
time. It forms paths in the constructed search "tree" instead of attaching relationships directly between parameters.

[5] - Similar to Optuna, Hyperopt supports TPE and Adaptive TPE, not sufficient to satisfy the "Multiple Algorithms"
category

[6] - Facebook's Ax tool requires `botorch` as well as some other heavy dependencies such as `sklearn` and `plotly`, 
highlighting another issue with these conglomerate libraries in which they slap technologies on top of each other 
without making them work together properly.

## Example Usage

### [Booth Function Optimization Test](https://www.sfu.ca/~ssurjano/booth.html)

<img src="https://render.githubusercontent.com/render/math?math=f(x) = (x_1 %2B 2x_2 - 7)^2 %2B (2x_1 %2B x_2 - 5)^2"></img>

<img src="https://render.githubusercontent.com/render/math?math=f(x) = 0, at x^* = (1, 3)"></img>

```python
In [1]: from blackboxopt import space

In [2]: from blackboxopt.algorithms import evolutionary as evo

In [3]: space.set_global_seed(42)

In [4]: space_dict = {'x1': space.RandFloat(-10, 10), 'x2': space.RandFloat(-10, 10)}

In [5]: sampler = space.SearchSpaceSampler(space_dict)

In [6]: def booth(x1, x2):
   ...:     return (x1 + 2*x2 - 7)**2 + (2*x1+x2-5)**2
   ...: 

In [7]: best_params = evo.genetic_algorithm(booth, sampler, maximize=False)
Generation 100: 100%|████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 533.43it/s]

In [8]: best_params
Out[8]: {'x1': 0.9668857125179429, 'x2': 3.105792305287739}

In [9]: booth(**best_params)
Out[9]: 0.03341694498219069

In [10]: space.set_global_seed(42)

In [11]: evo.genetic_algorithm(booth, sampler, maximize=False, pop_size=500)
Generation 100: 100%|████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 107.26it/s]
Out[11]: {'x1': 0.9913671375291351, 'x2': 3.019416302693932}

In [12]: space_dict = {'x1': space.RandInt(-10, 10), 'x2': space.RandInt(-10, 10)}

In [13]: sampler = space.SearchSpaceSampler(space_dict)

In [14]: evo.genetic_algorithm(booth, sampler, maximize=False)
Generation 100: 100%|████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 528.22it/s]
Out[14]: {'x1': 1, 'x2': 3}
```

### Conditional Search Spaces

```python
In [1]: from blackboxopt import space

In [2]: from blackboxopt.algorithms import evolutionary as evo

In [3]: space.set_global_seed(42)

In [4]: space_dict = {
    'x1': space.RandInt(-10, 10),
    'x2': space.Conditional(lambda sample: space.rng.integers(-11, sample['x1']), {'x1'})
}

In [5]: sampler = space.SearchSpaceSampler(space_dict)

In [6]: def booth(x1, x2):
   ...:     return (x1 + 2*x2 - 7)**2 + (2*x1+x2-5)**2
   ...: 

In [7]: best_params = evo.genetic_algorithm(booth, sampler, maximize=False)
Generation 100: 100%|██████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 500.87it/s]

In [8]: best_params
Out [8]: {'x1': 3, 'x2': 1}
```