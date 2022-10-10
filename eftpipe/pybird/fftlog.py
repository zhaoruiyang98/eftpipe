from __future__ import annotations
import numpy as np
from numpy import exp
from numpy import log
from numpy import pi
from numpy import sin

# from pyfftw.builders import rfft
from numpy.fft import rfft
from scipy.interpolate import CubicSpline


def CoefWindow(N, window: float | None = 1):
    """ FFTLog auxiliary function: window sending the FFT coefficients to 0 at the edges. From fast-pt """
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
    W[n[:] > n_right] = theta_right - 1 / (2 * pi) * sin(2 * pi * theta_right)
    W[n[:] < n_left] = theta_left - 1 / (2 * pi) * sin(2 * pi * theta_left)

    return W


class FFTLog(object):
    """
    A class implementing the FFTLog algorithm.

    Attributes
    ----------
    Nmax : int, optional
        maximum number of points used to discretize the function
    xmin : float, optional
        minimum of the function to transform
    xmax : float, optional
        maximum of the function to transform
    bias : float, optional
        power by which we modify the function as x**bias * f

    Methods
    -------
    setx()
        Calculates the discrete x points for the transform

    setPow()
        Calculates the power in front of the function

    Coef()
        Calculates the single coefficients

    sumCoefxPow(xin, f, x, window=1)
        Sums over the Coef * Pow reconstructing the input function
    """

    def __init__(self, **kwargs):
        self.Nmax = kwargs["Nmax"]
        if self.Nmax % 2 != 0:
            raise ValueError(f"expected even Nmax, instead of Nmax={self.Nmax}")
        self.xmin = kwargs["xmin"]
        self.xmax = kwargs["xmax"]
        self.bias = kwargs["bias"]
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
            self.Pow[i] = self.bias + 1j * 2.0 * pi / (self.Nmax * self.dx) * (
                i - self.Nmax / 2.0
            )

    def setCoefFactor(self):
        self._CoefFactor = self.xmin ** (-self.Pow) / float(self.Nmax)

    def Coef(
        self,
        xin,
        f,
        extrap="extrap",
        window: float | None = 1,
        log_interp: bool = False,
    ):
        """compute coefficients for FFTLog

        Parameters
        ----------
        xin : ndarray, 1d
            input x-axis data, values must be real, finite and in strictly increasing order
        f : ndarray
            input y-axis data, the last axis should match xin
        extrap : str, optional
            extrapolation mode, by default 'extrap'
        window : int, optional
            window parameter, by default 1
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
            _ = CubicSpline(xin, f, axis=-1, extrapolate=False)
            interpfunc = lambda x: _(np.log(x))

        _shape = list(f.shape)[:-1]
        fx = np.zeros(tuple(_shape + [self.Nmax]), dtype=np.float64)
        Coef = np.empty(tuple(_shape + [self.Nmax + 1]), dtype=complex)

        if extrap == "extrap":
            ileft = np.searchsorted(self.x, xin[0])
            iright = np.searchsorted(self.x, xin[-1], side="right")
            efactor = exp(-self.bias * np.arange(self.Nmax) * self.dx)
            fx[..., ileft:iright] = (
                interpfunc(self.x[ileft:iright]) * efactor[ileft:iright]
            )
            if xin[0] > self.x[0]:
                # print ('low extrapolation')
                nslow = (log(f[1]) - log(f[0])) / (log(xin[1]) - log(xin[0]))
                Aslow = f[0] / xin[0] ** nslow
                fx[..., 0:ileft] = Aslow * self.x[0:ileft] ** nslow * efactor[0:ileft]
            if xin[-1] < self.x[-1]:
                # print ('high extrapolation')
                nshigh = (log(f[-1]) - log(f[-2])) / (log(xin[-1]) - log(xin[-2]))
                Ashigh = f[-1] / xin[-1] ** nshigh
                fx[..., iright:] = Ashigh * self.x[iright:] ** nshigh * efactor[iright:]
        elif extrap == "padding":
            ileft = np.searchsorted(self.x, xin[0])
            iright = np.searchsorted(self.x, xin[-1], side="right")
            efactor = exp(-self.bias * np.arange(ileft, iright) * self.dx)
            fx[..., ileft:iright] = interpfunc(self.x[ileft:iright]) * efactor
        else:
            raise ValueError(f"unexpected extrap = {extrap}")

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
            fFFT[i] = np.real(np.sum(Coef * xi ** self.Pow))
        return fFFT
