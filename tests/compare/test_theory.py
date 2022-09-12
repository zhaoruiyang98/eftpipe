from __future__ import annotations
import pytest
from typing import Callable, Tuple
from typing_extensions import TypeAlias
from numpy import ndarray as NDArray

DiffbirdT: TypeAlias = Callable[[], Tuple[NDArray, NDArray]]


def has_pybird() -> bool:
    try:
        from pybird.pybird import Correlator
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not has_pybird(), reason="should install pybird")
def test_eftlss_vs_pybird(compare_ndarrays, diffbird: DiffbirdT):
    eftpipe_th, pybird_th = diffbird()
    eftpipe_dict = {k: v for k, v in zip(("P0", "P2", "P4"), eftpipe_th)}
    pybird_dict = {k: v for k, v in zip(("P0", "P2", "P4"), pybird_th)}
    compare_ndarrays(
        pybird_dict,
        eftpipe_dict,
        tolerances={
            "P0": {"atol": 1e-4, "rtol": 1e-4},
            "P2": {"atol": 1e-4, "rtol": 1e-4},
            "P4": {"atol": 1e-4, "rtol": 1e-4},
        },
    )
