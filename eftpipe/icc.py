"""Integral Constraint Corrections"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any, TYPE_CHECKING
from numpy import newaxis
from scipy.interpolate import interp1d
from cobaya.log import HasLogger
from .fftlog2d import FFTLog2D
from .fftlog2d import bessel_matrix
from .pybird.fftlog import FFTLog
from .pybird.pybird import common
from .window import MetaInfoError
from .window import window_kgrid
from .tools import replace_suffix
from .tools import root_only

if TYPE_CHECKING:
    from numpy import ndarray as NDArray
    from .pybird.pybird import Bird
    from .pybird.pybird import Common

try:
    import numba
except ImportError:

    def _lookup_table(x) -> dict[int, Any]:
        return {x: i for i, x in enumerate(x)}

    def _ICpannel_to_ndarray(arr):
        l1, l2, s1, s2 = (np.unique(_) for _ in arr.T[:4])
        create_table = lambda t: {x: i for i, x in enumerate(t)}
        table = {
            name: create_table(arr) for name, arr in zip(["l1", "l2", "s1", "s2"], [l1, l2, s1, s2])
        }
        ret = np.zeros(shape=(l1.size, l2.size, s1.size, s2.size), dtype="f8")
        for vl1, vl2, vs1, vs2, value in arr:
            i, j, k, l = (
                table["l1"][vl1],
                table["l2"][vl2],
                table["s1"][vs1],
                table["s2"][vs2],
            )
            ret[i, j, k, l] = value
        return ret

else:
    from numba import int64, float64
    from numba import njit
    from numba import prange

    @njit([(float64[:],), (int64[:],)], cache=True)
    def _lookup_table(x) -> dict[int, Any]:
        ret = dict()
        for i, v in enumerate(x):
            ret[v] = i
        return ret

    @njit(float64[:, :, :, :](float64[:, :]), parallel=True, cache=True)
    def _ICpannel_to_ndarray(arr):  # type: ignore
        l1 = np.unique(arr.T[0])
        l2 = np.unique(arr.T[1])
        s1 = np.unique(arr.T[2])
        s2 = np.unique(arr.T[3])
        l1t = _lookup_table(l1)
        l2t = _lookup_table(l2)
        s1t = _lookup_table(s1)
        s2t = _lookup_table(s2)
        ret = np.zeros(shape=(l1.size, l2.size, s1.size, s2.size), dtype=np.float64)
        for ii in prange(arr.shape[0]):
            vl1, vl2, vs1, vs2, value = arr[ii]
            i, j, k, l = l1t[vl1], l2t[vl2], s1t[vs1], s2t[vs2]
            ret[i, j, k, l] = value
        return ret


def read_configspace_IC_file(file: str | Path, info=print, warning=print):
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"File {file} does not exist")
    if file.suffix in (".txt", ".TXT", ".dat", ".ascii"):
        info("start reading configspace IC file")
        warning("reading csv file is very slow, please consider using npy file instead")
        try:
            import pandas as pd

            info("may take ~ 1 minutes")
            colnames = ["l1", "l2", "s1", "s2", "value"]
            df = pd.read_csv(file, names=colnames, delim_whitespace=True, engine="c", comment="#")
            ret = df.to_numpy()
        except ImportError:
            info("may take ~ 10 minutes")
            ret = np.loadtxt(file)
    elif file.suffix == ".npy":
        info("start reading configspace IC file")
        ret = np.load(file)
    else:
        raise ValueError(f"File {file} has unsupported suffix {file.suffix}")

    return ret


def ICpannel_to_ndarray(arr: NDArray, inorder=False, info=print):
    l1, l2, s1, s2 = (np.unique(_) for _ in arr.T[:4])
    assert l1.size * l2.size * s1.size * s2.size == arr.shape[0]
    meta = dict(l1=l1, l2=l2, s1=s1, s2=s2)

    if inorder:
        ret = arr.T[4].reshape((l1.size, l2.size, s1.size, s2.size))
    else:
        info("rearranging configspace IC array...")
        ret = _ICpannel_to_ndarray(arr)
    return ret, meta


class IntegralConstraint(HasLogger):
    r"""Integral Constraint Correction, ref: https://arxiv.org/abs/1904.08851

    Parameters
    ----------
    Pshot : float
        shot noise
    icc_fourier_file : str | Path
        path to fourier transformed icc npz file
        default None: always compute the icc matrix and will not save to disk
    icc_configspace_SN_file : str | Path
        path to configspace PSN file, the header should be like s, W0^2, W2^2, W4^2, ...
        default None: load file from disk
    icc_configspace_IC_file : str | Path
        path to configspace IC file, the header should be like l1, l2, s1, s2, value
        default None: load file from disk
    inorder : bool
        whether the value column of icc_configspace_IC_file is in order, default False
    co : Common
        this class only uses co.k and co.Nk, if Na, Nl not specified, use co.Nl
        default pybird.Common
    load : bool
        set it True to load fourier file from disk, default True
    save : bool
        set it True to save fourier file to disk, default True
    check_meta : bool
        set it True to check meta info when loading fourier file, default True
        if meta info not exists, report deprecated warning (will raise Exception in the future),
        if meta info not match, raise MetaInfoError
    Na : int
        output multipoles, default co.Nl
    Nl : int
        input multipoles, default co.Nl
    pmax : float
        integrate window up to pmax, default 0.3
    accboost : int
        accboost x times of p grids, default 1
    withmask : bool
        set it True to mask window, default True
    windowk : float
        mask width, default 0.05, i.e. when k=0.1, only p in [0.1 - 0.05, 0.1 + 0.05] will be integrated
    Nmax : int
        FFTLog nodes for PSN, default 4096
    bias : float
        FFTLog bias for PSN, default -2.1
    window_param : float | None
        FFTLog window parameter for PSN, default 1
    Nxmax : int
        FFTLog nodes for Wal, x-axis, default 4096
    Nymax : int
        FFTLog nodes for Wal, y-axis, default 4096
    xbias : float
        FFTLog bias for Wal, x-axis, default -2.0
    ybias : float
        FFTLog bias for Wal, y-axis, default -2.0
    windowxy_param : float | None
        FFTLog window parameter for Wal, default 1
    name : str
        logger name, default 'eftpipe.icc'
    snapshot : bool
        take snapshot after applying ``icc``, by default False

    Math
    ----
    The integral constraint correction (configuration space) for eBOSS DR16 ELG is given by:

    ..math::
        -\left(P_{\mathrm{shot}}\mathcal{W}_{\ell_{1}}^{\mathrm{sn}}(s_{1})+\sum_{\ell_{2},s_{2}}\frac{4\pi s_{2}^{2}ds_{2}}{2\ell_{2}+1}\mathcal{W}_{\ell_{1},\ell_{2}}^{\mathrm{ic}}(s_{1},s_{2})\xi_{\ell_{2}}(s_{2})\right)

    this formula applies to window-convolved correlation function,
    where the first term can be directly FFTLog to fourier space,
    and the second term is window-like convolution in fourier space, ie.

    ..math::
        P_{\ell}^{\mathrm{ic}}(k) = -\sum_{\ell'}\int dk'\;W_{\ell,\ell'}^{\mathrm{ic}}(k,k')P_{\ell'}(k')

    where the window matrix can be evaluated via FFTLog2D

    ..math::
        W_{\ell,\ell'}^{\mathrm{ic}}(k,k') = \frac{8(-i)^{\ell}i^{\ell'}}{2\ell'+1}k'^{2}\int ds_{1}\int ds_{2}\;s_{1}^{2}s_{2}^{2}\mathcal{W}_{\ell,\ell'}^{\mathrm{ic}}(s_{1},s_{2})j_{\ell}(ks_{1})j_{\ell'}(k's_{2})

    """

    def __init__(
        self,
        Pshot: float,
        icc_fourier_file: str | Path | None = None,
        icc_configspace_SN_file: str | Path | None = None,
        icc_configspace_IC_file: str | Path | None = None,
        inorder: bool = False,
        co: Common = common,
        load: bool = True,
        save: bool = True,
        check_meta: bool = True,
        Na: int | None = None,
        Nl: int | None = None,
        pmax: float = 0.3,
        accboost: int = 1,
        withmask: bool = True,
        windowk: float = 0.05,
        Nmax: int = 4096,
        bias: float = -2.1,
        window_param: float | None = 1,
        Nxmax: int = 4096,
        Nymax: int = 4096,
        xbias: float = -2.0,
        ybias: float = -2.0,
        windowxy_param: float | None = 1,
        name: str = "eftpipe.icc",
        snapshot: bool = False,
    ) -> None:
        self.set_logger(name=name)
        self.co = co
        if all(
            _ is None
            for _ in (
                icc_fourier_file,
                icc_configspace_SN_file,
                icc_configspace_IC_file,
            )
        ):
            raise ValueError("No ICC file specified")

        def resolve_or_None(path: str | Path | None):
            return Path(path).resolve() if path else None

        self.icc_fourier_file = resolve_or_None(icc_fourier_file)
        self.icc_configspace_SN_file = resolve_or_None(icc_configspace_SN_file)
        self.icc_configspace_IC_file = resolve_or_None(icc_configspace_IC_file)
        self.inorder = inorder
        self._load = load
        self._save = save if self.icc_fourier_file else False
        self.check_meta = check_meta
        self._create_meta = True
        self.withmask = withmask
        self.windowk = windowk
        Na = Na or self.co.Nl
        Nl = Nl or self.co.Nl
        if Na > self.co.Nl or Nl > self.co.Nl:
            raise ValueError(
                f"request Na={Na}, Nl={Nl} while bird only compute Nl up to {self.co.Nl}"
            )
        if Na > Nl:
            raise ValueError(f"dangerous settings Na={Na} > Nl={Nl}")
        self.p = window_kgrid(kmax=pmax, accboost=accboost)

        def str_or_None(x):
            return str(x) if x else None

        self.meta: dict[str, Any] = dict(
            Na=Na,
            Nl=Nl,
            pmax=pmax,
            accboost=accboost,
            Nmax=Nmax,
            bias=bias,
            window_param=window_param,
            Nxmax=Nxmax,
            Nymax=Nymax,
            xbias=xbias,
            ybias=ybias,
            windowxy_param=windowxy_param,
            icc_configspace_SN_file=str_or_None(self.icc_configspace_SN_file),
            icc_configspace_IC_file=str_or_None(self.icc_configspace_IC_file),
            k=self.co.k.tolist(),
        )
        self.Pshot = Pshot
        self.PSN, self.Wal = self._loadicc()
        if self.PSN is None:
            self.PSN = self._compute_PSN()
            self.Wal = self._compute_Wal()
        self.Waldk = self._compute_Waldk()
        if self._save:
            self._saveicc()
        self.PSN *= Pshot  # always need Pshot
        self.snapshot = snapshot
        if self.snapshot:
            self.mpi_info("snapshot is enabled")

    def _loadicc(self) -> tuple[NDArray, NDArray] | tuple[None, None]:
        """load fourier matrix, return None if failed"""
        PSN, Wal = None, None
        icc_fourier_file = self.icc_fourier_file
        if self._load:
            try:
                data = np.load(icc_fourier_file)  # type: ignore
                PSN, Wal = data["PSN"], data["Wal"]
                self.mpi_info("Loaded icc from %s", icc_fourier_file)
            except (OSError, TypeError):
                self.mpi_warning("Cannot load icc from %s", icc_fourier_file)
            else:
                # retry
                if Wal.shape[1] != self.meta["Nl"]:
                    PSN, Wal = None, None
                    self.mpi_info(
                        "Nl not match, load the one with suffix '_Nl%d'",
                        self.meta["Nl"],
                    )
                    assert icc_fourier_file
                    self.icc_fourier_file = replace_suffix(
                        icc_fourier_file, f"_Nl{self.meta['Nl']}.npz"
                    )
                    icc_fourier_file = self.icc_fourier_file
                    try:
                        data = np.load(icc_fourier_file)
                        PSN, Wal = data["PSN"], data["Wal"]
                        self.mpi_info("Loaded icc from %s", icc_fourier_file)
                    except OSError:
                        self.mpi_warning("Cannot load icc from %s", icc_fourier_file)
            if (PSN is not None) and self.check_meta:
                assert icc_fourier_file
                meta_file = replace_suffix(icc_fourier_file, ".json")
                if not meta_file.exists():
                    self.mpi_warning(
                        "Deprecated: when check_meta=True and load=True, "
                        "IntegralConstraint will try to find the meta-info "
                        "json file and compare it to current settings. "
                        "In the future, if the meta-info file does not exists, "
                        "an Exception will be raised."
                    )
                    self._create_meta = False
                else:
                    with meta_file.open("r") as f:
                        meta = json.load(f)
                    # copy meta info from previous file
                    if self.meta["icc_configspace_SN_file"] is None:
                        self.meta["icc_configspace_SN_file"] = meta["icc_configspace_SN_file"]
                    if self.meta["icc_configspace_IC_file"] is None:
                        self.meta["icc_configspace_IC_file"] = meta["icc_configspace_IC_file"]
                    if meta != self.meta:
                        raise MetaInfoError(
                            f"inconsistent meta info\n"
                            f"loaded icc's meta:\n{meta}\n"
                            f"expect:\n{self.meta}"
                        )
        if PSN is not None:
            self._save = False

        return PSN, Wal

    def _compute_PSN(self):
        self.mpi_info("Computing PSN")
        self.mpi_warning(
            "please manually check the PSN after computation, "
            "since the default FFTLog settings may not be optimal"
        )
        Na: int = self.meta["Na"]
        if self.icc_configspace_SN_file is None:
            raise ValueError("please specify icc_configspace_SN_file")

        try:
            data = np.loadtxt(self.icc_configspace_SN_file)
            # remove s = 0
            while data[0, 0] == 0.0:
                data = data[1:, :]
            data = data[:, : 1 + Na]
        except OSError:
            raise OSError(
                f"Error: cannot load icc_configspace_SN_file from {self.icc_configspace_SN_file}"
            )
        except IndexError as ex:
            raise TypeError(f"loaded icc_configspace_SN_file has unexpected shape") from ex

        s = data[:, 0]
        xi = data[:, 1:].T
        fft = FFTLog(
            Nmax=self.meta["Nmax"],
            xmin=s[0],
            xmax=s[-1],
            bias=self.meta["bias"],
        )
        # a, n
        coef = fft.Coef(s, xi, extrap="padding", window=self.meta["window_param"])
        # k, n
        power = self.co.k[:, newaxis] ** (-fft.Pow[newaxis, :] - 3.0)
        # a, n
        mat = np.array([bessel_matrix(fft.Pow, ell) for ell in range(0, 2 * Na, 2)])
        PSN: NDArray = np.einsum("an,kn,an->ak", coef, power, mat, optimize=True).real
        PSN *= 4 * np.pi * np.array([(-1j) ** ell for ell in range(0, 2 * Na, 2)]).real[:, newaxis]
        return PSN

    def _compute_Wal(self):
        self.mpi_info("Computing new ic Wal")
        self.mpi_warning(
            "please manually check the ic Wal after computation, "
            "since the default FFTLog settings may not be optimal"
        )
        Na: int = self.meta["Na"]
        Nl: int = self.meta["Nl"]
        if self.icc_configspace_IC_file is None:
            raise ValueError("please specify icc_configspace_IC_file")

        data = read_configspace_IC_file(
            self.icc_configspace_IC_file, self.mpi_info, self.mpi_warning
        )
        data, pannel_meta = ICpannel_to_ndarray(data, self.inorder, self.mpi_info)
        s1in, s2in = pannel_meta["s1"], pannel_meta["s2"]
        fft2d = FFTLog2D(
            Nxmax=self.meta["Nxmax"],
            Nymax=self.meta["Nymax"],
            xmin=10**-3,
            xmax=s1in[-1],
            ymin=10**-3,
            ymax=s2in[-1],
            xbias=self.meta["xbias"],
            ybias=self.meta["ybias"],
        )
        Wal = np.empty((Na, Nl, self.co.k.size, self.p.size))
        for il1 in range(Na):
            for il2 in range(Nl):
                Wal[il1, il2, ...] = fft2d.spherical_transform(
                    s1in,
                    s2in,
                    data[il1, il2, ...],
                    extrap="padding",
                    window=self.meta["windowxy_param"],
                    k1=self.co.k,
                    k2=self.p,
                    l1=2 * il1,
                    l2=2 * il2,
                )
                Wal[il1, il2, ...] *= (
                    8.0
                    * np.real((-1j) ** (2 * il1) * (1j) ** (2 * il2))
                    / (2 * (2 * il2) + 1)
                    * self.p**2
                )
        return Wal

    def _compute_Waldk(self):
        assert self.Wal is not None
        Wal_masked = self.Wal
        if self.withmask:
            pgrid, kgrid = np.meshgrid(self.p, self.co.k, indexing="ij")
            mask = (pgrid > kgrid - self.windowk) & (pgrid < kgrid + self.windowk)
            Wal_masked = np.einsum("alkp,pk->alkp", self.Wal, mask)

        # the spacing (need to do the convolution as a sum)
        deltap = self.p[1:] - self.p[:-1]  # type: ignore
        deltap = np.concatenate([[0], deltap])
        return np.einsum("alkp,p->alkp", Wal_masked, deltap)

    @root_only
    def _saveicc(self):
        assert self.icc_fourier_file
        self.mpi_info("Saving PSN and Wal to %s", self.icc_fourier_file)
        np.savez(self.icc_fourier_file, PSN=self.PSN, Wal=self.Wal)  # type: ignore
        if self._create_meta:
            meta_file = replace_suffix(self.icc_fourier_file, ".json")
            with meta_file.open("w") as f:
                json.dump(self.meta, f, indent=2)

    def integrWindow(self, P):
        """
        Convolve the window functions to a power spectrum P
        """
        Pk = interp1d(
            self.co.k,
            P,
            axis=-1,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",  # type: ignore
        )(self.p)
        # (multipole l, multipole ' p, k, k' m) , (multipole ', power pectra s, k' m)
        return np.einsum("alkp,lsp->ask", self.Waldk, Pk, optimize=True)

    def icc(self, bird: Bird):
        self.mpi_warning("This method is wrong and deprecated, please don't use it!")
        bird.P11l -= self.integrWindow(bird.P11l)
        bird.Pctl -= self.integrWindow(bird.Pctl)
        bird.Ploopl -= self.integrWindow(bird.Ploopl)
        if bird.co.with_NNLO:
            bird.PctNNLOl -= self.integrWindow(bird.PctNNLOl)
        bird.Pstl -= self.integrWindow(bird.Pstl)
        assert self.PSN is not None
        bird.Picc -= self.PSN
        if self.snapshot:
            bird.create_snapshot("icc")
