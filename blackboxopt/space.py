from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

rng = np.random.default_rng()


def set_global_seed(seed):
    global rng
    rng = np.random.default_rng(seed)


class Space(ABC):
    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError


class RandInt(Space):
    def __init__(self, low: int, high: Optional[int] = None):
        if high is None:
            self.low = 0
            self.high = high
        else:
            self.low = low
            self.high = high

    def sample(self, *args, **kwargs):
        return rng.integers(self.low, self.high + 1)


class RandFloat(Space):
    def __init__(self, low: float, high: Optional[float] = None):
        if high is None:
            self.low = 0.
            self.high = low
        else:
            self.low = low
            self.high = high

    def sample(self, *args, **kwargs):
        return rng.uniform(self.low, self.high)


class RandUniform(RandFloat):
    def sample(self, *args, **kwargs):
        return rng.uniform(self.low, self.high)


class RandLogUniform(RandUniform):
    def __init__(self, low: float, high: Optional[float] = None, base: Optional[float] = np.e):
        super(RandLogUniform, self).__init__(low, high)
        self.base = base

    def sample(self, *args, **kwargs):
        return np.power(self.base, super(RandLogUniform, self).sample())


class Discrete(Space):
    def __init__(self, choices: Iterable[Any]):
        self.choices = list(choices)

    def sample(self, *args, **kwargs):
        return rng.choice(self.choices)


class Conditional(Space):
    def __init__(self, func: Callable[[Dict[str, Any]], Any]):
        self.func = func

    def sample(self, assignments, *args, **kwargs):
        return self.func(assignments)


SearchSpace = Dict[str, Space]
