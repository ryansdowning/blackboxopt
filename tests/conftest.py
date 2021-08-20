import pytest

from blackboxopt import space


@pytest.fixture(autouse=True)
def set_seed():
    space.set_global_seed(42)
