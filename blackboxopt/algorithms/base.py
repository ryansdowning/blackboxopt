from typing import Callable, Union, Dict, Any, Tuple
from abc import ABC, abstractmethod
from blackboxopt.space import SearchSpace
import inspect


def handle_base_params(
        func: Callable[..., float], space: SearchSpace, maximize: bool = True
) -> Tuple[Callable[..., float], SearchSpace]:
    assert set(inspect.getfullargspec(func)[0]) == set(space.keys())
    if maximize:
        return func, space
    else:
        return lambda *args, **kwargs: -func(*args, **kwargs), space


class Algorithm(ABC):
    def __init__(self, func: Callable[..., float], space: SearchSpace, *args, **kwargs):
        assert set(inspect.getfullargspec(func)[0]) == set(space.keys())
        self.func = func
        self.space = space

    @abstractmethod
    def optimize(self) -> Dict[str, Any]:
        raise NotImplementedError


class RandomAlgorithm(Algorithm):
    def __init__(self, func: Callable[..., float], space: SearchSpace):
        super(RandomAlgorithm, self).__init__(func, space)

    def optimize(self) -> Dict[str, Any]:
        return {k: v.sample() for k, v in self.space.items()}
