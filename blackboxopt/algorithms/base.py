import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple, Union

from blackboxopt.space import SearchSpace


def handle_base_params(
        func: Callable[..., float], space: SearchSpace, maximize: bool = True
) -> Tuple[Callable[..., float], SearchSpace]:
    assert set(inspect.getfullargspec(func)[0]) == set(space.keys())
    if maximize:
        return func, space
    else:
        return lambda *args, **kwargs: -func(*args, **kwargs), space


def random_algorithm(func: Callable[..., float], space: SearchSpace, maximize: bool = True):
    func, space = handle_base_params(func, space, maximize)
    return {k: v.sample() for k, v in space.items()}
