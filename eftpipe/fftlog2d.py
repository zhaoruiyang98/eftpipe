from __future__ import annotations

import numpy as np
from typing import TypeVar
from numpy import ndarray as NDArray
from numpy import newaxis
from numpy import pi
from scipy.interpolate import interp2d
from scipy.special import loggamma

_ComplexOrArrayT = TypeVar("_ComplexOrArrayT", complex, NDArray)


class FFTLog2D:
    """2D FFTLog algorithm"""

    def __init__(
        self,
        Nxmax: int,
        Nymax: int,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        xbias: float,
        ybias: float,
    ) -> None:
        self.Nxmax, self.Nymax = Nxmax, Nymax
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.xbias, self.ybias = xbias, ybias

        self.dx: float = np.log(self.xmax / self.xmin) / (self.Nxmax - 1)
        self.dy: float = np.log(self.ymax / self.ymin) / (self.Nymax - 1)
        self.x: NDArray = np.geomspace(
            self.xmin, self.xmax, self.Nxmax, dtype=np.float64
        )
        self.y: NDArray = np.geomspace(
            self.ymin, self.ymax, self.Nymax, dtype=np.float64
        )
        self.xPow: NDArray = (
            self.xbias + 2 * pi * 1j * np.fft.fftfreq(self.Nxmax, d=self.dx)
        )
        self.yPow: NDArray = (
            self.ybias + 2 * pi * 1j * np.fft.fftfreq(self.Nymax, d=self.dy)
        )

    def Coef(
        self,
        xin: NDArray,
        yin: NDArray,
        zin: NDArray,
        extrap: str = "padding",
        nf: int | None = None,
    ) -> NDArray:

        f = interp2d(xin, yin, zin, kind="cubic")
        farr = np.zeros(shape=(self.Nxmax, self.Nymax), dtype=np.float64)

        if extrap == "extrap":
            raise NotImplementedError  # I'm lazy...
        elif extrap == "padding":
            maskx = np.logical_and(self.x >= xin[0], self.x <= xin[-1])
            masky = np.logical_and(self.y >= yin[0], self.y <= yin[-1])
            mask2d = np.outer(maskx, masky)
            farr[mask2d] = f(self.x[maskx], self.y[masky]).reshape(-1)
        else:
            raise ValueError("extra should be 'extrap' or 'padding'")

        out = (
            1
            / (self.Nxmax * self.Nymax)
            * np.fft.fft2(
                farr
                * np.outer(
                    np.power(self.x / self.x[0], -self.xbias),
                    np.power(self.y / self.y[0], -self.ybias),
                )
            )
            / np.outer(np.power(self.x[0], self.xPow), np.power(self.y[0], self.yPow))
        )

        if nf is not None:
            out *= self.window(nf)

        return out

    def spherical_transform(
        self,
        xin: NDArray,
        yin: NDArray,
        zin: NDArray,
        extrap: str = "padding",
        nf: int | None = None,
        *,
        k1: NDArray,
        k2: NDArray,
        l1: int,
        l2: int,
    ) -> NDArray:

        Coef = self.Coef(xin, yin, zin, extrap=extrap, nf=nf)  # m, n
        k1grid, xpowgrid = k1[:, newaxis], (-3.0 - self.xPow)[newaxis, :]
        # p, m broadcasting
        M1 = np.power(k1grid, xpowgrid) * self.bessel_matrix_function(self.xPow, l1)
        k2grid, ypowgrid = k2[:, newaxis], (-3.0 - self.yPow)[newaxis, :]
        # q, n broadcasting
        M2 = np.power(k2grid, ypowgrid) * self.bessel_matrix_function(self.yPow, l2)
        return np.real(np.einsum("mn,pm,qn->pq", Coef, M1, M2, optimize=True))

    def bessel_matrix_function(self, p: _ComplexOrArrayT, l: int) -> _ComplexOrArrayT:
        r"""helper function to do bessel transfrom

        Parameters
        ----------
        p : ndarray | complex
            power
        l : int
            order of spherical bessel function

        Returns
        -------
        ndarray | complex

        Notes
        -----
        This function evaluates the following integral:

        .. math::
            \int_0^\infty s_1^{2+p} j_l(s_1)\;ds_1
        
        k dependence can be obtained by multiplying  the factor :math:`k^{-3-p}`
        """
        out = np.exp(loggamma(0.5 * (3.0 + l + p)) - loggamma(0.5 * (l - p)))
        return out * np.power(2.0, 1.0 + p) * np.sqrt(pi)

    def window(self, nf: int) -> NDArray:
        fx = np.fft.fftfreq(self.Nxmax, d=1.0)
        fy = np.fft.fftfreq(self.Nymax, d=1.0)
        fx_left, fx_right = fx[-nf], fx[nf]
        fy_left, fy_right = fy[-nf], fy[nf]
        fx_min = np.min(fx)
        fx_max = -fx_min  # standard order for fft
        fy_min = np.min(fy)
        fy_max = -fy_min

        thetax_left = (fx[fx < fx_left] - fx_min) / (fx_left - fx_min)
        thetax_right = (fx_max - fx[fx > fx_right]) / (fx_max - fx_right)
        thetay_left = (fy[fy < fy_left] - fy_min) / (fy_left - fy_min)
        thetay_right = (fy_max - fy[fy > fy_right]) / (fy_max - fy_right)

        wx = np.ones(self.Nxmax)
        wy = np.ones(self.Nymax)
        wx[fx < fx_left] = thetax_left - 1 / (2 * pi) * np.sin(2 * pi * thetax_left)
        wx[fx > fx_right] = thetax_right - 1 / (2 * pi) * np.sin(2 * pi * thetax_right)
        wy[fy < fy_left] = thetay_left - 1 / (2 * pi) * np.sin(2 * pi * thetay_left)
        wy[fy > fy_right] = thetay_right - 1 / (2 * pi) * np.sin(2 * pi * thetay_right)

        return np.outer(wx, wy)
