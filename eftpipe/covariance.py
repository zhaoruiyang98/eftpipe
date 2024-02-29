from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .typing import ndarrayf

# integration of the product of four legendre polynomials
# \int_{-1}^{1} d\mu L_a(\mu) L_b(\mu) L_c(\mu) L_d(\mu)
Gabcd: dict[tuple[int, int, int, int], float] = {
    (0, 0, 0, 0): 2,
    (0, 0, 0, 2): 0,
    (0, 0, 0, 4): 0,
    (0, 0, 2, 0): 0,
    (0, 0, 2, 2): (2 / 5),
    (0, 0, 2, 4): 0,
    (0, 0, 4, 0): 0,
    (0, 0, 4, 2): 0,
    (0, 0, 4, 4): (2 / 9),
    (0, 2, 0, 0): 0,
    (0, 2, 0, 2): (2 / 5),
    (0, 2, 0, 4): 0,
    (0, 2, 2, 0): (2 / 5),
    (0, 2, 2, 2): (4 / 35),
    (0, 2, 2, 4): (4 / 35),
    (0, 2, 4, 0): 0,
    (0, 2, 4, 2): (4 / 35),
    (0, 2, 4, 4): (40 / 693),
    (0, 4, 0, 0): 0,
    (0, 4, 0, 2): 0,
    (0, 4, 0, 4): (2 / 9),
    (0, 4, 2, 0): 0,
    (0, 4, 2, 2): (4 / 35),
    (0, 4, 2, 4): (40 / 693),
    (0, 4, 4, 0): (2 / 9),
    (0, 4, 4, 2): (40 / 693),
    (0, 4, 4, 4): (36 / 1001),
    (2, 0, 0, 0): 0,
    (2, 0, 0, 2): (2 / 5),
    (2, 0, 0, 4): 0,
    (2, 0, 2, 0): (2 / 5),
    (2, 0, 2, 2): (4 / 35),
    (2, 0, 2, 4): (4 / 35),
    (2, 0, 4, 0): 0,
    (2, 0, 4, 2): (4 / 35),
    (2, 0, 4, 4): (40 / 693),
    (2, 2, 0, 0): (2 / 5),
    (2, 2, 0, 2): (4 / 35),
    (2, 2, 0, 4): (4 / 35),
    (2, 2, 2, 0): (4 / 35),
    (2, 2, 2, 2): (6 / 35),
    (2, 2, 2, 4): (24 / 385),
    (2, 2, 4, 0): (4 / 35),
    (2, 2, 4, 2): (24 / 385),
    (2, 2, 4, 4): (3578 / 45045),
    (2, 4, 0, 0): 0,
    (2, 4, 0, 2): (4 / 35),
    (2, 4, 0, 4): (40 / 693),
    (2, 4, 2, 0): (4 / 35),
    (2, 4, 2, 2): (24 / 385),
    (2, 4, 2, 4): (3578 / 45045),
    (2, 4, 4, 0): (40 / 693),
    (2, 4, 4, 2): (3578 / 45045),
    (2, 4, 4, 4): (40 / 1001),
    (4, 0, 0, 0): 0,
    (4, 0, 0, 2): 0,
    (4, 0, 0, 4): (2 / 9),
    (4, 0, 2, 0): 0,
    (4, 0, 2, 2): (4 / 35),
    (4, 0, 2, 4): (40 / 693),
    (4, 0, 4, 0): (2 / 9),
    (4, 0, 4, 2): (40 / 693),
    (4, 0, 4, 4): (36 / 1001),
    (4, 2, 0, 0): 0,
    (4, 2, 0, 2): (4 / 35),
    (4, 2, 0, 4): (40 / 693),
    (4, 2, 2, 0): (4 / 35),
    (4, 2, 2, 2): (24 / 385),
    (4, 2, 2, 4): (3578 / 45045),
    (4, 2, 4, 0): (40 / 693),
    (4, 2, 4, 2): (3578 / 45045),
    (4, 2, 4, 4): (40 / 1001),
    (4, 4, 0, 0): (2 / 9),
    (4, 4, 0, 2): (40 / 693),
    (4, 4, 0, 4): (36 / 1001),
    (4, 4, 2, 0): (40 / 693),
    (4, 4, 2, 2): (3578 / 45045),
    (4, 4, 2, 4): (40 / 1001),
    (4, 4, 4, 0): (36 / 1001),
    (4, 4, 4, 2): (40 / 1001),
    (4, 4, 4, 4): (1058 / 17017),
}


@dataclass(repr=False, eq=False)
class Multipole:
    P0: ndarrayf
    P2: ndarrayf
    P4: ndarrayf = field(default=None)  # type: ignore

    def __post_init__(self):
        if self.P4 is None:
            self.P4 = np.zeros_like(self.P0)
        if self.P0.shape != self.P2.shape != self.P4.shape:
            raise ValueError("P0, P2, P4 should have the same shape")


def parse_subscripts(subscripts: str):
    import re

    pattern = re.compile(r"([a-zA-Z0-9]{2})((,[a-zA-Z0-9]{2})*)->([a-zA-Z0-9]{4})")
    subscripts = "".join(subscripts.split())  # remove whitespace
    match = pattern.match(subscripts)
    if not match:
        raise ValueError("invalid subscripts")
    op1, ops, _, result = match.groups()
    ops = ops or ","
    ops = [op1, *(_ for _ in ops[1:].split(",") if _)]

    # parity even
    def normalize(s):
        return "".join(sorted(s))

    ops = [normalize(_) for _ in ops]
    a, b, c, d = result
    requires = [normalize(_) for _ in (a + c, b + d, a + d, b + c)]
    for x in requires:
        if x not in ops:
            raise ValueError(f"invalid subscripts: input multipole {x} is missing")
    indices = tuple(ops.index(_) for _ in requires)
    return indices


class GaussianCovariance:
    """Gaussian Covariance matrix for simulations"""

    kedges: ndarrayf
    thin_shell: bool

    def __init__(
        self,
        kedges,
        nmodes=None,
        volume: float | None = None,
        kavg=None,
        thin_shell: bool = True,
    ):
        """
        Parameters
        ----------
        kedges : array_like
            Edges of the k-bins.
        volume : float, optional
            Volume of the simulation box in the unit of (Gpc/h)^3.
        nmodes : array_like, optional
            Number of modes in each k-bin. If not given, estimated from volume.
        kavg : array_like, optional
            Averaged k in each k-bin. If not given, estimated from kedges.
            Only affect the case when thin_shell=True.
        thin_shell : bool
            use thin shell approximation, default is False
        """
        if not thin_shell:
            raise NotImplementedError
        self.thin_shell = thin_shell
        self.kedges = np.asarray(kedges, dtype=np.float64)
        self._kavg = np.asarray(kavg, dtype=np.float64) if kavg is not None else None

        if nmodes is not None:
            self.nmodes = nmodes
        else:
            if volume is None:
                raise ValueError("either nmodes or volume should be given")
            self._nmodes = volume * 1e9 * self.kvolume / (2 * np.pi) ** 3

    @property
    def nmodes(self):
        return self._nmodes

    @nmodes.setter
    def nmodes(self, value):
        nmodes = np.asarray(value, dtype=np.float64)
        if nmodes.size != self.kedges.size - 1:
            raise ValueError("nmodes should have length kedges.size - 1")
        self._nmodes = nmodes

    @property
    def kvolume(self):
        lower, upper = self.kedges[:-1], self.kedges[1:]
        return (4 * np.pi / 3) * (upper**3 - lower**3)

    @property
    def kavg(self):
        if self._kavg is not None:
            return self._kavg
        lower, upper = self.kedges[:-1], self.kedges[1:]
        return (3 / 4) * (upper**4 - lower**4) / (upper**3 - lower**3)

    def fullcov(
        self,
        Paa: Multipole,
        Pbb: Multipole,
        Pab: Multipole,
        ells: list[int] = [0, 2, 4],
    ):
        aaaa = self(ells, ells, "aa->aaaa", Paa)
        aabb = self(ells, ells, "aa,bb,ab->aabb", Paa, Pbb, Pab)
        aaab = self(ells, ells, "aa,bb,ab->aaab", Paa, Pbb, Pab)
        bbbb = self(ells, ells, "bb->bbbb", Pbb)
        bbba = self(ells, ells, "aa,bb,ab->bbba", Paa, Pbb, Pab)
        abab = self(ells, ells, "aa,bb,ab->abab", Paa, Pbb, Pab)
        # fmt: off
        return np.block(
            [[aaaa, aabb, aaab],
             [aabb, bbbb, bbba],
             [aaab, bbba, abab]]
        )
        # fmt: on

    def __call__(self, ell1, ell2, subscripts: str, /, *multipoles: Multipole):
        """
        Parameters
        ----------
        ell1 : int or array_like
            multipole index
        ell2 : int or array_like
            multipole index
        subscripts : str
            specifies the subscripts for covariance matrix calculation.
            For example, "aa,bb,ab->abbb" means computing Cov[P_ab, P_bb]
            using P_aa, P_bb and P_ab as input multipoles.
        multipoles : Multipole
            input multipoles, **SHOULD INCLUDE SHOTNOISE!**

        Returns
        -------
        1d ndarray if ell1 and ell2 are both int, otherwise 2d ndarray
        """
        try:
            Pac, Pbd, Pad, Pbc = (multipoles[i] for i in parse_subscripts(subscripts))
        except IndexError:
            raise ValueError("missing input multipole")
        if any(_.size != self.kedges.size - 1 for _ in (Pac.P0, Pbd.P0, Pad.P0, Pbc.P0)):
            raise ValueError("input multipoles have incorrect shape")
        ell1, ell2 = np.atleast_1d(ell1), np.atleast_1d(ell2)
        db = {}
        for l1, l2 in ((x, y) for x in ell1 for y in ell2):
            key = tuple(sorted((l1, l2)))
            if key in db:
                # symmetry
                continue
            # fmt: off
            sigmak = (
                (2 * l1 + 1) * (2 * l2 + 1) / 2
                * (
                    (Pac.P0 * Pbd.P0 + Pad.P0 * Pbc.P0) * Gabcd[(l1, l2, 0, 0)]
                    + (Pac.P2 * Pbd.P2 + Pad.P2 * Pbc.P2) * Gabcd[(l1, l2, 2, 2)]
                    + (Pac.P4 * Pbd.P4 + Pad.P4 * Pbc.P4) * Gabcd[(l1, l2, 4, 4)]
                    + (Pac.P0 * Pbd.P2 + Pad.P0 * Pbc.P2) * Gabcd[(l1, l2, 0, 2)]
                    + (Pac.P2 * Pbd.P0 + Pad.P2 * Pbc.P0) * Gabcd[(l1, l2, 2, 0)]
                    + (Pac.P0 * Pbd.P4 + Pad.P0 * Pbc.P4) * Gabcd[(l1, l2, 0, 4)]
                    + (Pac.P4 * Pbd.P0 + Pad.P4 * Pbc.P0) * Gabcd[(l1, l2, 4, 0)]
                    + (Pac.P2 * Pbd.P4 + Pad.P2 * Pbc.P4) * Gabcd[(l1, l2, 2, 4)]
                    + (Pac.P4 * Pbd.P2 + Pad.P4 * Pbc.P2) * Gabcd[(l1, l2, 4, 2)]
                )
            )
            # fmt: on
            db[key] = sigmak / self.nmodes
        if len(ell1) == len(ell2) == 1:
            return db[ell1[0], ell2[0]]
        toret = []
        for l1 in ell1:
            tmp = []
            for l2 in ell2:
                key = tuple(sorted((l1, l2)))
                tmp.append(np.diag(db[key]))
            toret.append(tmp)
        return np.block(toret)


if __name__ == "__main__":
    # examples
    kedges = np.linspace(0, 0.3, 30 + 1)
    volume = 2.0**3
    gcov = GaussianCovariance(kedges, volume=volume)
    Paa = Multipole(
        P0=10000 * np.random.random(kedges.size - 1) + 1 / 1e-4,
        P2=10000 * np.random.random(kedges.size - 1),
        P4=10000 * np.random.random(kedges.size - 1),
    )
    Pbb = Multipole(
        P0=1000 * np.random.random(kedges.size - 1) + 1 / 1e-4,
        P2=1000 * np.random.random(kedges.size - 1),
        P4=1000 * np.random.random(kedges.size - 1),
    )
    Pab = Multipole(
        P0=4000 * np.random.random(kedges.size - 1) + 0,
        P2=4000 * np.random.random(kedges.size - 1),
        P4=4000 * np.random.random(kedges.size - 1),
    )
    # diagonal term
    sigma_00_aa = gcov(0, 0, "aa->aa", Paa)
    sigma_02_abba = gcov(0, 2, "aa,bb,ab->abba", Paa, Pbb, Pab)
    # single covariance matrix
    cov_aaaa = gcov([0, 2, 4], [0, 2, 4], "aa->aaaa", Paa)
    # full covariance matrix
    cov_all = gcov.fullcov(Paa, Pbb, Pab)
