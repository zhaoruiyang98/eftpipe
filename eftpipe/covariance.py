from __future__ import annotations
import argparse
import logging
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING
from eftpipe.analysis import Multipole

if TYPE_CHECKING:
    from eftpipe.etyping import ndarrayf

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


@dataclass
class MultipoleMeasurement:
    shotnoise: float
    k: ndarrayf
    mono: ndarrayf
    quad: ndarrayf
    hex: ndarrayf

    @classmethod
    def from_multipole(cls, multipole: Multipole, shotnoise: float = 0):
        hex = multipole.get("hex")
        if hex is None:
            hex = np.zeros(multipole.k.size)
        else:
            hex = hex.to_numpy()
        return cls(
            shotnoise=shotnoise,
            k=multipole.k.copy(),
            mono=multipole.mono().to_numpy(),
            quad=multipole.quad().to_numpy(),
            hex=hex,
        )

    def __repr__(self) -> str:
        return (
            f"MultipoleMeasurement(shotnoise={self.shotnoise:.3e}, "
            f"kmin={self.k[0]:.3f}, "
            f"kmax={self.k[-1]:.3f})"
        )


def cov_XXXX(l: int, ll: int, mXX: MultipoleMeasurement):
    P0, P2, P4, SN = mXX.mono, mXX.quad, mXX.hex, mXX.shotnoise
    P0 = P0 + SN
    coef = (2 * l + 1) * (2 * ll + 1)
    cov = coef * (
        P0**2 * Gabcd[(0, 0, l, ll)]
        + 2 * P0 * P2 * Gabcd[(0, 2, l, ll)]
        + P2**2 * Gabcd[(2, 2, l, ll)]
        + 2 * P0 * P4 * Gabcd[(0, 4, l, ll)]
        + 2 * P2 * P4 * Gabcd[(2, 4, l, ll)]
        + P4**2 * Gabcd[(4, 4, l, ll)]
    )
    return np.diag(cov)


def cov_YYYY(l: int, ll: int, mYY: MultipoleMeasurement):
    return cov_XXXX(l, ll, mYY)


def cov_XXYY(l: int, ll: int, mXY: MultipoleMeasurement):
    P0, P2, P4 = mXY.mono, mXY.quad, mXY.hex
    coef = (2 * l + 1) * (2 * ll + 1)
    cov = coef * (
        P0**2 * Gabcd[(0, 0, l, ll)]
        + 2 * P0 * P2 * Gabcd[(0, 2, l, ll)]
        + P2**2 * Gabcd[(2, 2, l, ll)]
        + 2 * P0 * P4 * Gabcd[(0, 4, l, ll)]
        + 2 * P2 * P4 * Gabcd[(2, 4, l, ll)]
        + P4**2 * Gabcd[(4, 4, l, ll)]
    )
    return np.diag(cov)


def cov_XXXY(l: int, ll: int, mXX: MultipoleMeasurement, mXY: MultipoleMeasurement):
    P0X, P2X, P4X, SN = mXX.mono, mXX.quad, mXX.hex, mXX.shotnoise
    P0X = P0X + SN
    P0XY, P2XY, P4XY = mXY.mono, mXY.quad, mXY.hex
    coef = (2 * l + 1) * (2 * ll + 1)
    cov = coef * (
        P0X * P0XY * Gabcd[(0, 0, l, ll)]
        + P2X * P2XY * Gabcd[(2, 2, l, ll)]
        + (P0X * P2XY + P2X * P0XY) * Gabcd[(0, 2, l, ll)]
        + P4X * P4XY * Gabcd[(4, 4, l, ll)]
        + (P0X * P4XY + P4X * P0XY) * Gabcd[(0, 4, l, ll)]
        + (P2X * P4XY + P4X * P2XY) * Gabcd[(2, 4, l, ll)]
    )
    return np.diag(cov)


def cov_YYYX(l: int, ll: int, mYY: MultipoleMeasurement, mXY: MultipoleMeasurement):
    return cov_XXXY(l, ll, mYY, mXY)


def cov_XYXY(
    l: int,
    ll: int,
    mXX: MultipoleMeasurement,
    mYY: MultipoleMeasurement,
    mXY: MultipoleMeasurement,
):
    P0X, P2X, P4X, SNX = mXX.mono, mXX.quad, mXX.hex, mXX.shotnoise
    P0X = P0X + SNX
    P0Y, P2Y, P4Y, SNY = mYY.mono, mYY.quad, mYY.hex, mYY.shotnoise
    P0Y = P0Y + SNY
    P0XY, P2XY, P4XY = mXY.mono, mXY.quad, mXY.hex
    coef = (2 * l + 1) * (2 * ll + 1) * 1 / 2
    cov = coef * (
        (P0XY**2 + P0X * P0Y) * Gabcd[(0, 0, l, ll)]
        + (P0X * P2Y + P0Y * P2X + 2 * P0XY * P2XY) * Gabcd[(0, 2, l, ll)]
        + (P2XY**2 + P2X * P2Y) * Gabcd[(2, 2, l, ll)]
        + (P0X * P4Y + P0Y * P4X + 2 * P0XY * P4XY) * Gabcd[(0, 4, l, ll)]
        + (P2X * P4Y + P2Y * P4X + 2 * P2XY * P4XY) * Gabcd[(2, 4, l, ll)]
        + (P4XY**2 + P4X * P4Y) * Gabcd[(4, 4, l, ll)]
    )
    return np.diag(cov)


def gaussian_covariance(
    volume: float,
    kbin_width: float,
    PXX: MultipoleMeasurement,
    PYY: MultipoleMeasurement | None = None,
    PXY: MultipoleMeasurement | None = None,
):
    """gaussian covariance in a simulation box

    Notes
    -----
    P11, P22 and P12 should share the same k bins.
    """
    is_multi_tracer: bool = (PYY is not None) and (PXY is not None)
    if is_multi_tracer:
        k = (PXX.k + PYY.k + PXY.k) / 3  # type: ignore
    else:
        k = PXX.k
    if PYY is None or PXY is None:
        k = PXX.k
    else:
        k = (PXX.k + PYY.k + PXY.k) / 3
    number_of_k_modes = (
        4 * np.pi * k**2 * kbin_width / ((2 * np.pi) / volume ** (1 / 3)) ** 3
    )
    nkinv = np.diag(1 / number_of_k_modes)
    nkinv = np.block(
        [[nkinv, nkinv, nkinv], [nkinv, nkinv, nkinv], [nkinv, nkinv, nkinv]]
    )
    CXXXX = np.block(
        [
            [cov_XXXX(0, 0, PXX), cov_XXXX(0, 2, PXX), cov_XXXX(0, 4, PXX)],
            [cov_XXXX(2, 0, PXX), cov_XXXX(2, 2, PXX), cov_XXXX(2, 4, PXX)],
            [cov_XXXX(4, 0, PXX), cov_XXXX(4, 2, PXX), cov_XXXX(4, 4, PXX)],
        ]
    )
    if not is_multi_tracer:
        return CXXXX * nkinv
    assert PYY is not None and PXY is not None
    # fmt: off
    CYYYY = np.block(
        [
            [cov_YYYY(0, 0, PYY), cov_YYYY(0, 2, PYY), cov_YYYY(0, 4, PYY)],
            [cov_YYYY(2, 0, PYY), cov_YYYY(2, 2, PYY), cov_YYYY(2, 4, PYY)],
            [cov_YYYY(4, 0, PYY), cov_YYYY(4, 2, PYY), cov_YYYY(4, 4, PYY)],
        ]
    )
    CXYXY = np.block(
        [
            [cov_XYXY(0, 0, PXX, PYY, PXY), cov_XYXY(0, 2, PXX, PYY, PXY), cov_XYXY(0, 4, PXX, PYY, PXY)],
            [cov_XYXY(2, 0, PXX, PYY, PXY), cov_XYXY(2, 2, PXX, PYY, PXY), cov_XYXY(2, 4, PXX, PYY, PXY)],
            [cov_XYXY(4, 0, PXX, PYY, PXY), cov_XYXY(4, 2, PXX, PYY, PXY), cov_XYXY(4, 4, PXX, PYY, PXY)],
        ]
    )
    CXXYY = np.block(
        [
            [cov_XXYY(0, 0, PXY), cov_XXYY(0, 2, PXY), cov_XXYY(0, 4, PXY)],
            [cov_XXYY(2, 0, PXY), cov_XXYY(2, 2, PXY), cov_XXYY(2, 4, PXY)],
            [cov_XXYY(4, 0, PXY), cov_XXYY(4, 2, PXY), cov_XXYY(4, 4, PXY)],
        ]
    )
    CXXXY = np.block(
        [
            [cov_XXXY(0, 0, PXX, PXY), cov_XXXY(0, 2, PXX, PXY), cov_XXXY(0, 4, PXX, PXY)],
            [cov_XXXY(2, 0, PXX, PXY), cov_XXXY(2, 2, PXX, PXY), cov_XXXY(2, 4, PXX, PXY)],
            [cov_XXXY(4, 0, PXX, PXY), cov_XXXY(4, 2, PXX, PXY), cov_XXXY(4, 4, PXX, PXY)],
        ]
    )
    CYYYX = np.block(
        [
            [cov_YYYX(0, 0, PYY, PXY), cov_YYYX(0, 2, PYY, PXY), cov_YYYX(0, 4, PYY, PXY)],
            [cov_YYYX(2, 0, PYY, PXY), cov_YYYX(2, 2, PYY, PXY), cov_YYYX(2, 4, PYY, PXY)],
            [cov_YYYX(4, 0, PYY, PXY), cov_YYYX(4, 2, PYY, PXY), cov_YYYX(4, 4, PYY, PXY)],
        ]
    )
    nkinv = np.block(
        [[nkinv, nkinv, nkinv], [nkinv, nkinv, nkinv], [nkinv, nkinv, nkinv]]
    )
    cov = np.block(
        [[  CXXXX,   CXXYY, CXXXY],
         [CXXYY.T,   CYYYY, CYYYX],
         [CXXXY.T, CYYYX.T, CXYXY]]
    )
    # fmt: on
    return cov * nkinv


def main():
    parser = argparse.ArgumentParser(
        description="Compute the gaussian covariance matrix for a given set of multipoles."
    )
    parser.add_argument("--PXX", required=True, help="path to XX multipole file")
    parser.add_argument("--PYY", help="path to YY multipole file")
    parser.add_argument("--PXY", help="path to XY multipole file")
    parser.add_argument(
        "--volume", type=float, required=True, help="volume of the box, in Gpc^3"
    )
    parser.add_argument("--kbin", type=float, required=True, help="k bin width")
    parser.add_argument("--SNXX", type=float, required=True, help="shotnoise of XX")
    parser.add_argument("--SNYY", type=float, help="shotnoise of YY")
    parser.add_argument("-o", "--output", required=True, help="output file")
    args = parser.parse_args()

    is_multi_tracer = args.PYY and args.PXY
    if is_multi_tracer and args.SNYY is None:
        parser.error("--SNYY is required for multi-tracer covariance")

    # setup logger
    logger = logging.getLogger(parser.prog)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(name)s: %(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if not is_multi_tracer:
        PXX = MultipoleMeasurement.from_multipole(
            Multipole.loadtxt(args.PXX, logger=logger), args.SNXX
        )
        cov = gaussian_covariance(args.volume * 10**9, args.kbin, PXX)
        np.savetxt(args.output, cov)
        return
    PXX = MultipoleMeasurement.from_multipole(
        Multipole.loadtxt(args.PXX, logger=logger), args.SNXX
    )
    PYY = MultipoleMeasurement.from_multipole(
        Multipole.loadtxt(args.PYY, logger=logger), args.SNYY
    )
    PXY = MultipoleMeasurement.from_multipole(
        Multipole.loadtxt(args.PXY, logger=logger)
    )
    cov = gaussian_covariance(args.volume * 10**9, args.kbin, PXX, PYY, PXY)
    np.savetxt(args.output, cov)


if __name__ == "__main__":
    main()
