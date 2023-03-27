from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from scipy.special import legendre
from .pybird.pybird import BirdLike
from .transformer import BirdTransformer

if TYPE_CHECKING:
    from .typing import ndarrayf


def chain_coeff(l: int) -> float:
    R"""compute ``A_\ell`` coeff for chained power spectrum

    Parameters
    ----------
    l : int

    Returns
    -------
    float

    Notes
    -----
    .. math:: \frac{(2\ell+1)\mathcal{L}_{\ell}(0)}{(2\ell+5)\mathcal{L}_{\ell+2}(0)}
    """
    return ((2 * l + 1) * legendre(l)(0)) / ((2 * l + 5) * legendre(l + 2)(0))


class Chained(BirdTransformer):
    def chained_matrix(self, Nl: int):
        # fmt: off
        if Nl == 2:
            return np.array(
                [[1, -chain_coeff(0)]],
                dtype=np.float64,
            )
        elif Nl == 3:
            return np.array(
                [[1, -chain_coeff(0),               0],
                 [0,               1, -chain_coeff(2)],],
                dtype=np.float64,
            )
        elif Nl == 4:
            return np.array(
                [[1, -chain_coeff(0),               0,               0],
                 [0,               1, -chain_coeff(2),               0],
                 [0,               0,               1, -chain_coeff(4)]],
                dtype=np.float64,
            )
        else:
            raise NotImplementedError
        # fmt: on

    def transform(self, birdlike: BirdLike):
        mat = self.chained_matrix(birdlike.co.Nl)
        to_chained = lambda P: np.einsum("al,l...->a...", mat, P, optimize=True)
        self.f = birdlike.f
        self.co = birdlike.co
        self.P11l = to_chained(birdlike.P11l)
        self.Ploopl = to_chained(birdlike.Ploopl)
        self.Pctl = to_chained(birdlike.Pctl)
        self.PctNNLOl = to_chained(birdlike.PctNNLOl)
        self.Pstl = to_chained(birdlike.Pstl)
        self.Picc = to_chained(birdlike.Picc)
        return self
