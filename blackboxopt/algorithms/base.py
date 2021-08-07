"""Module contains helper functions are other basic features for implementing optimization algorithms"""
import inspect
from typing import Any, Callable, Dict, Tuple

from blackboxopt.space import SearchSpaceSampler


def handle_base_params(
    func: Callable[..., float], sampler: SearchSpaceSampler, maximize: bool = True
) -> Tuple[Callable[..., float], SearchSpaceSampler]:
    """Helper function which verifies the base arguments of the optimization algorithm (commonly used in
    `blackboxopt.algorithms.evolutionary`) are valid, and normalizes functions so optimization is always maximized

    Args:
        func: The function which is being optimized
        sampler: The SearchSpaceSampler being use to generate samples for the function being optimized
        maximize: Flag signalling if the function is being maximized or minimized, default True: maximized

    Returns:
        func and space where the func was inverted if it was minimized, thus standardizing it to a maximization problem
    """
    assert set(inspect.getfullargspec(func)[0]) == set(sampler.space.keys())
    if maximize:
        return func, sampler
    return lambda *args, **kwargs: -func(*args, **kwargs), sampler


def random_algorithm(
    func: Callable[..., float], sampler: SearchSpaceSampler, maximize: bool = True
) -> Dict[str, Any]:
    """Dummy optimization algorithm that just returns a random choice of parameter values

    Args:
        func: The function which is being optimized
        sampler: The SearchSpaceSampler being use to generate samples for the function being optimized
        maximize: Flag signalling if the function is being maximized or minimized, default True: maximized

    Returns:
        Dictionary of parameter values chosen by the algorithm (random)
    """
    _, _ = handle_base_params(func, sampler, maximize)
    return sampler.sample()
