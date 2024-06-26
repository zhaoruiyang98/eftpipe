from __future__ import annotations
import numpy as np
from numpy import exp
from numpy import log
from numpy import pi
from numpy import sin

# from pyfftw.builders import rfft
from numpy.fft import rfft
from scipy.interpolate import CubicSpline


def one(x):
    return 1


def CoefWindow(N, window=1, left=True, right=True):
    """FFTLog auxiliary function: window sending the FFT coefficients to 0 at the edges. From fast-pt"""
    n = np.arange(-N // 2, N // 2 + 1)
    if window == 1:
        n_cut = N // 2
    else:
        n_cut = int(window * N // 2.0)

    n_right = n[-1] - n_cut
    n_left = n[0] + n_cut

    n_r = n[n[:] > n_right]
    n_l = n[n[:] < n_left]

    theta_right = (n[-1] - n_r) / float(n[-1] - n_right - 1)
    theta_left = (n_l - n[0]) / float(n_left - n[0] - 1)

    W = np.ones(n.size)
    if right:
        W[n[:] > n_right] = theta_right - 1 / (2 * pi) * sin(2 * pi * theta_right)
    if left:
        W[n[:] < n_left] = theta_left - 1 / (2 * pi) * sin(2 * pi * theta_left)

    return W


class FFTLog(object):
    """
    A class implementing the FFTLog algorithm.

    Parameters
    ----------
    Nmax : int
        maximum number of points used to discretize the function
    xmin : float
        lower bound of transformation range
    xmax : float
        upper bound of transformation range
    bias : float
        power by which we modify the function as x**bias * f
    """

    def __init__(self, Nmax, xmin, xmax, bias):
        self.Nmax = Nmax
        if self.Nmax % 2 != 0:
            raise ValueError(f"expected even Nmax, instead of Nmax={self.Nmax}")
        self.xmin = xmin
        self.xmax = xmax
        self.bias = bias
        self.dx = np.log(self.xmax / self.xmin) / (self.Nmax - 1.0)
        self.setx()
        self.setPow()
        self.setCoefFactor()

    def setx(self):
        self.x = np.empty(self.Nmax)
        for i in range(self.Nmax):
            self.x[i] = self.xmin * np.exp(i * self.dx)

    def setPow(self):
        self.Pow = np.empty(self.Nmax + 1, dtype=complex)
        for i in range(self.Nmax + 1):
            self.Pow[i] = self.bias + 1j * 2.0 * pi / (self.Nmax * self.dx) * (i - self.Nmax / 2.0)

    def setCoefFactor(self):
        self._CoefFactor = self.xmin ** (-self.Pow) / float(self.Nmax)

    def Coef(self, xin, f, extrap="extrap", window=1, log_interp=False, kernel=one):
        """compute coefficients for FFTLog

        Parameters
        ----------
        xin : ndarray, 1d
            input x-axis data, values must be real, finite and in strictly increasing order
        f : ndarray
            input y-axis data, the last axis should match xin
        extrap : str or tuple[str, str], optional
            extrapolation mode, by default 'extrap'
        window : float, optional
            window parameter, by default 1; cutting off highest frequencies if None
        log_interp : bool, optional
            do interpolation in log-x scale, by default False

        Returns
        -------
        ndarray

        Raises
        ------
        ValueError
            if extrap is not 'extrap' or 'padding'

        Notes
        -----
        when doing exponential extrapolation, ns may not be properly computed or the extrapolated f is not damping
        """
        if not log_interp:
            interpfunc = CubicSpline(xin, f, axis=-1, extrapolate=False)
        else:
            _interpfunc = CubicSpline(np.log(xin), f, axis=-1, extrapolate=False)
            interpfunc = lambda x: _interpfunc(np.log(x))
        if not isinstance(extrap, tuple):
            extrap = (extrap,) * 2
        if (
            (kernel is not one)
            and ("extrap" in extrap)
            and (xin[0] > self.x[0] and xin[-1] < self.x[-1])
        ):
            raise ValueError("kernel is not supported in 'extrap' mode")

        _shape = f.shape[:-1]
        fx = np.zeros(_shape + (self.Nmax,), dtype=np.float64)
        Coef = np.empty(_shape + (self.Nmax + 1,), dtype=complex)

        if any(_ not in ["padding", "extrap"] for _ in extrap):
            raise ValueError(f"unexpected extrap = {extrap}")
        ileft = np.searchsorted(self.x, xin[0])
        iright = np.searchsorted(self.x, xin[-1], side="right")
        xtrunc = self.x[ileft:iright]
        efactor = exp(-self.bias * np.arange(ileft, iright) * self.dx)
        if kernel is not one:
            efactor = efactor * kernel(xtrunc)
        fx[..., ileft:iright] = interpfunc(xtrunc) * efactor
        if extrap[0] == "extrap" and xin[0] > self.x[0]:
            # low extrapolation
            nslow = (log(f[1]) - log(f[0])) / (log(xin[1]) - log(xin[0]))
            Aslow = f[0] / xin[0] ** nslow
            efactor = exp(-self.bias * np.arange(0, ileft) * self.dx)
            fx[..., 0:ileft] = Aslow * self.x[0:ileft] ** nslow * efactor
        if extrap[-1] == "extrap" and xin[-1] < self.x[-1]:
            # high extrapolation
            nshigh = (log(f[-1]) - log(f[-2])) / (log(xin[-1]) - log(xin[-2]))
            Ashigh = f[-1] / xin[-1] ** nshigh
            efactor = exp(-self.bias * np.arange(iright, self.Nmax) * self.dx)
            fx[..., iright:] = Ashigh * self.x[iright:] ** nshigh * efactor

        tmp = rfft(fx, axis=-1)  # numpy
        # tmp = rfft(fx, planner_effort='FFTW_ESTIMATE')() ### pyfftw

        Coef[..., : self.Nmax // 2] = np.conj(tmp[..., 1:][..., ::-1])
        Coef[..., self.Nmax // 2 :] = tmp[..., :]
        Coef *= self._CoefFactor

        if window is not None:
            Coef *= CoefWindow(self.Nmax, window=window)
        else:
            Coef[..., 0] /= 2.0
            Coef[..., self.Nmax] /= 2.0

        return Coef

    def sumCoefxPow(self, xin, f, x, window=1):
        Coef = self.Coef(xin, f, window=window)
        fFFT = np.empty_like(x)
        for i, xi in enumerate(x):
            fFFT[i] = np.real(np.sum(Coef * xi**self.Pow))
        return fFFT
