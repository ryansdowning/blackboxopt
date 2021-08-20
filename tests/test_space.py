import numpy as np
import pytest

from blackboxopt import space


def test_set_rng_seed(monkeypatch):
    def mock_default_rng(seed):
        return seed
    monkeypatch.setattr(np.random, 'default_rng', mock_default_rng)
    space.set_global_seed(42)
    assert space.rng == 42


def test_rand_bool(monkeypatch):
    bool_space = space.RandBool()
    assert isinstance(bool_space.sample(), bool)
    assert repr(bool_space) == 'RandBool()'
    assert [bool_space.sample() for _ in range(5)] == [True, False, False, True, False]

    class MockGenerator:
        @staticmethod
        def random():
            return 0
    monkeypatch.setattr(space, 'rng', MockGenerator)
    assert bool_space.sample() is True

    class MockGenerator:
        @staticmethod
        def random():
            return 1
    monkeypatch.setattr(space, 'rng', MockGenerator)
    assert bool_space.sample() is False


def test_rand_int(monkeypatch):
    int_space = space.RandInt(-10, 10)
    assert isinstance(int_space.sample(), np.int_)
    assert repr(int_space) == 'RandInt(low=-10, high=10)'
    assert [int_space.sample() for _ in range(5)] == [6, 3, -1, -1, 8]

    int_space2 = space.RandInt(10)
    assert int_space2.low == 0
    assert int_space2.high == 10
    assert repr(int_space2) == 'RandInt(low=0, high=10)'

    class MockGenerator:
        @staticmethod
        def integers(low, high):
            return low
    monkeypatch.setattr(space, 'rng', MockGenerator)
    assert int_space.sample() == -10

    class MockGenerator:
        @staticmethod
        def integers(low, high):
            return high - 1
    monkeypatch.setattr(space, 'rng', MockGenerator)
    assert int_space.sample() == 10

    with pytest.raises(AssertionError):
        space.RandInt(10, 0)


def test_rand_float(monkeypatch):
    float_space = space.RandFloat(-10., 10.)
    assert isinstance(float_space.sample(), float)
    assert repr(float_space) == 'RandFloat(low=-10.0, high=10.0)'
    assert [float_space.sample() for _ in range(5)] == [
        -1.2224312049589532, 7.171958398227648, 3.9473605811872776, -8.11645304224701, 9.512447032735118
    ]

    float_space2 = space.RandFloat(10.)
    assert float_space2.low == 0.
    assert float_space2.high == 10.
    assert repr(float_space2) == 'RandFloat(low=0.0, high=10.0)'

    class MockGenerator:
        @staticmethod
        def uniform(low, high):
            return low
    monkeypatch.setattr(space, 'rng', MockGenerator)
    assert float_space.sample() == -10.

    class MockGenerator:
        @staticmethod
        def uniform(low, high):
            return high
    monkeypatch.setattr(space, 'rng', MockGenerator)
    assert float_space.sample() == 10.

    with pytest.raises(AssertionError):
        space.RandFloat(10, 0)


def test_rand_log_uniform(monkeypatch):
    log_space = space.RandLogUniform(1e3, 1e6, base=10)
    assert isinstance(log_space.sample(), float)
    assert np.isclose(log_space.log_low, 3)
    assert np.isclose(log_space.log_high, 6)
    assert repr(log_space) == f'RandLogUniform(low=1000.0, high=1000000.0, base=10)'
    assert np.allclose(
        [log_space.sample() for _ in range(5)],
        [20731.7192636945, 376524.9501831178, 123624.36879527065, 1916.6024706981702, 845020.130213905]
    )

    log_space2 = space.RandLogUniform(np.e**20)
    assert np.isclose(log_space2.low, np.e)
    assert np.isclose(log_space2.high, np.e**20)
    assert np.isclose(log_space2.log_low, 1.)
    assert np.isclose(log_space2.log_high, 20.)

    class MockGenerator:
        @staticmethod
        def uniform(low, high):
            return low
    monkeypatch.setattr(space, 'rng', MockGenerator)
    assert np.isclose(log_space.sample(), 1e3)

    class MockGenerator:
        @staticmethod
        def uniform(low, high):
            return high
    monkeypatch.setattr(space, 'rng', MockGenerator)
    assert np.isclose(log_space.sample(), 1e6)

    with pytest.raises(AssertionError):
        space.RandLogUniform(100, 10, base=10)

    with pytest.raises(AssertionError):
        space.RandLogUniform(0, 1000, base=10)

    with pytest.raises(AssertionError):
        space.RandLogUniform(-1000, 1000, base=10)


def test_rand_discrete(monkeypatch):
    str_choices = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    discrete_space = space.RandDiscrete(str_choices)
    assert isinstance(discrete_space.sample(), str)
    assert repr(discrete_space) == f'RandDiscrete(choices={str_choices})'
    assert [discrete_space.sample() for _ in range(5)] == ['h', 'g', 'e', 'e', 'i']

    discrete_space2 = space.RandDiscrete(range(10))
    assert discrete_space2.choices == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert repr(discrete_space2) == 'RandDiscrete(choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])'

    class MockGenerator:
        @staticmethod
        def choice(choices):
            return choices[0]
    monkeypatch.setattr(space, 'rng', MockGenerator)
    assert discrete_space.sample() == 'a'
    assert discrete_space2.sample() == 0

    class MockGenerator:
        @staticmethod
        def choice(choices):
            return choices[-1]
    monkeypatch.setattr(space, 'rng', MockGenerator)
    assert discrete_space.sample() == 'j'
    assert discrete_space2.sample() == 9
