"""Space module implements the abstract search space representation that is used in the optimization algorithms.
Provides functionality for sampling from the search space and creating dependency relationships between parameters."""
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, TypeVar

import numpy as np

# pylint: disable=R0903, C0103, W0603
rng = np.random.default_rng()


def set_global_seed(seed):
    """Globally sets/resets the numpy RNG seed

    Args:
        seed: Integer, numpy seed to use
    """
    global rng
    rng = np.random.default_rng(seed)


class Space(ABC):
    """Abstract Base Class for search spaces"""

    @abstractmethod
    def sample(self, *args, **kwargs) -> Any:
        """Generates a sample from the search space

        All implementations must accept *args and **kwargs in order to be compatible with each other
        """
        raise NotImplementedError

    def __repr__(self):
        """Creates a nice string representation of the search space and its settings"""
        return f"""{self.__class__.__name__}({', '.join(f"{k}={v}" for k, v in vars(self).items())})"""


class RandBool(Space):
    """Search space for a random boolean"""

    def sample(self, *args, **kwargs):
        """Generates a random boolean with 50% probability of True and 50% probability of False"""
        return rng.random() <= 0.5


class RandInt(Space):
    """Search space for a random integer within the given range (inclusive)"""

    def __init__(self, low: int, high: Optional[int] = None):
        """
        Args:
            low: integer representing the lower bound of the space
            high: integer representing the upper bound of the space
        """
        if high is None:
            self.low, self.high = 0, low
        else:
            self.low, self.high = low, high
        assert self.low <= self.high, f"low cannot be greater than high for the sample range, " \
                                      f"got {self.low=} and {self.high=}"

    def sample(self, *args, **kwargs):
        """Generates a random integer within the specified range by using numpy's Generator.integers"""
        return rng.integers(self.low, self.high + 1)


class RandFloat(Space):
    """Search space for a random float within the given range (inclusive)"""

    def __init__(self, low: float, high: Optional[float] = None):
        """
        Args:
            low: float representing the lower bound of the space
            high: float representing the upper bound of the space
        """
        if high is None:
            self.low, self.high = 0.0, low
        else:
            self.low, self.high = low, high
        assert self.low <= self.high, f"low cannot be greater than high for the sample range, " \
                                      f"got {self.low=} and {self.high=}"

    def sample(self, *args, **kwargs):
        """Generates a random (uniform) float within the specified range by using numpy's Generator.uniform"""
        return rng.uniform(self.low, self.high)


class RandLogUniform(RandFloat):
    """Search space for a random float where the range was transformed to log space, sample is drawn from the log space
    and the base is applied with the sample as exponent
    """

    def __init__(
        self, low: float, high: Optional[float] = None, base: Optional[float] = np.e
    ):
        """
        Args:
            low: float representing the lower bound of the space (before applying base)
            high: float representing the upper bound of the space (before applying base)
            base: float representing the log base to apply to the sample, default natural log
        """
        if high is None:
            low, high = base, low
        super().__init__(low, high)
        assert self.low > 0, f"low must be greater than 0 for RandLogUniform, got {low}"
        assert self.high > 0, f"high must be greater than 0 for RandLogUniform, got {high}"
        self.base = base
        self.log_low = math.log(self.low, base)
        self.log_high = math.log(self.high, base)

    def sample(self, *args, **kwargs):
        """Generates a random float sample with numpy's Generator.uniform and applies base with sample as exponent"""
        return self.base**rng.uniform(self.log_low, self.log_high)

    def __repr__(self):
        return f"{self.__class__.__name__}(low={self.low}, high={self.high}, base={self.base})"


class RandDiscrete(Space):
    """Search space for a random discrete value"""

    def __init__(self, choices: Iterable[Any]):
        """
        Args:
            choices: Iterable of discrete possible choices.
            **Should probably be all of the same type, but this is not enforced**
        """
        self.choices = list(choices)

    def sample(self, *args, **kwargs):
        """Generate a random discrete choice using numpy's Generator.choice"""
        return rng.choice(self.choices)


class Conditional(Space):
    """A conditional search space is a sample space that depends on the result of other parameters
    (i.e. three parameters are sampled where the third must be between the two)"""

    def __init__(self, func: Callable[[Dict[str, Any]], Any], depends_on: Set[str]):
        """
        Args:
            func: The conditional function which produces a sample when a dictionary of the other parameter values
                  it depends on is provided
            depends_on: A set of the parameter names which this conditional space needs for sampling
        """
        if len(depends_on) == 0:
            raise AttributeError(
                "Conditional space was specified but does not depend on any other parameters?"
            )
        self.func = func
        self.depends_on = depends_on

    def sample(self, *args, assignments=None, **kwargs):
        """Generates a sample from the conditional space by passing the dependent assignments to the conditional's
        generation function"""
        if assignments is None or self.depends_on != self.depends_on & set(
            assignments.keys()
        ):
            raise ValueError(
                f"Conditional depends on {self.depends_on} but was only given {assignments=}"
            )
        return self.func(assignments)


T = TypeVar("T")


def solve_dependency_order(dependency_graph: Dict[T, Set[T]]) -> Tuple[T]:
    """Generic dependency resolver using topological sort to determine the appropriate order of dependencies

    Args:
        dependency_graph: An adjacency matrix where the keys are dependencies and the values are the dependencies which
                          must come before them

    Returns:
        Tuple of the dependencies where the order of the elements represents the resolved order of dependency execution
    """
    dependencies = set(dependency_graph.keys())
    dependency_order = []
    while dependencies:  # Resolve dependencies as they are available
        has_no_remaining_dep = set(
            filter(lambda dep: not dependency_graph[dep], dependencies)
        )
        # If there are no dependencies that can be resolved, then the dependency cannot be solved, cycle exists.
        if not has_no_remaining_dep:
            raise ValueError(
                f"Could not resolve the remaining dependencies: {dependencies}\n"
                f"There is probably a cycle in the dependencies which is impossible to solve."
            )
        for feature in has_no_remaining_dep:
            dependency_order.append(feature)
            dependencies.remove(feature)
            # After resolving dependency, remove it from the dependencies of all other features
            for feat in dependency_graph:
                dependency_graph[feat].discard(feature)
    return tuple(dependency_order)


SearchSpace = Dict[str, Space]


class SearchSpaceSampler:
    """Combines SearchSpaces to create a sampler that can process dependency relationships between the different
    search space parameters"""

    def __init__(self, space_dict: SearchSpace):
        """
        Args:
            space_dict: Dictionary where the keys are the names of the parameters and the values are SearchSpace objects
                        representing those parameters
        """
        self.space = space_dict

        self._conditionals = dict()
        for param, sp in self.space.items():
            if isinstance(sp, Conditional):
                self._conditionals[param] = sp

        dependency_graph = dict(
            (param, sp.depends_on.copy()) if isinstance(sp, Conditional) else (param, set())
            for param, sp in self.space.items()
        )
        self.sample_order = solve_dependency_order(dependency_graph)
        self.curr_sample = {param: None for param in self.sample_order}

    def sample(self) -> Dict[str, Any]:
        """Generates a sample of all parameters in the Sampler, returning a dictionary where the keys are the parameters
        names and the values are the value of each parameter for the generated sample"""
        samples = dict()
        for param in self.sample_order:
            samp = self.space[param].sample(assignments=samples)
            samples[param] = samp
            self.curr_sample[param] = samp
        return samples

    def resample_one(self, param):
        """Uses the cached sample (most recent sample) to generate a new sample value for a single parameter in the
        search space

        Args:
            param: Name of parameter to resample for

        Returns:
            Resampled value for the given parameter
        """
        assert param in self.space, f"Parameter: {param} is not part of this SearchSpaceSampler " \
                                    f"with parameters: {self.sample_order}"
        self.curr_sample.pop(param)
        samp = self.space[param].sample(assignments=self.curr_sample)
        self.curr_sample[param] = samp
        return samp

    def __repr__(self):
        """Creates string for rebuilding the SearchSpaceSampler object"""
        return f"""{self.__class__.__name__}({{{', '.join(f"'{k}': {v}" for k, v in self.space.items())}}}"""
