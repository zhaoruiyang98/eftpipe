from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast, NamedTuple, TYPE_CHECKING, Union
from pathlib import Path
from cobaya.log import HasLogger
from scipy.interpolate import interp1d
from scipy.special import spherical_jn
from numpy import newaxis
from numpy.typing import NDArray
from .tools import replace_suffix
from .tools import root_only
from .pybird.fftlog import FFTLog
from .pybird.pybird import Common, common, MPC

if TYPE_CHECKING:
    from typing_extensions import TypeAlias
    from .icc import IntegralConstraint
    from .pybird.pybird import Bird
    from .etyping import ndarrayf

Location: TypeAlias = Union[str, Path]


def window_kgrid(kmax: float = 0.3, accboost: int = 1) -> NDArray:
    return np.concatenate(
        [
            np.geomspace(1e-5, 0.015, 100 * accboost, endpoint=False),
            np.arange(0.015, kmax, 1e-3 / accboost),
        ]
    )


class MetaInfoError(Exception):
    pass


class Window(HasLogger):
    R"""Window effect

    Parameters
    ----------
    window_fourier_file: str | Path
        path to fourier transformed window matrix
        default None: always recompute the window and will not save to disk
    window_configspace_file: str | Path
        path to configspace window file, the header should be s, Q0, Q2, Q4, ...
        default None: load fourier matrix from disk
    co: Common
        this class only uses co.k and co.Nk, if Na, Nl not specified, use co.Nl
        default pybird.Common
    load: bool
        set it True to load fourier matrix from disk, default True
    save: bool
        set it True to save fourier matrix to disk, default True
    check_meta: bool
        set it True to check meta info when loading fourier matrix, default True
        if meta info not exists, report deprecated warning (will raise Exception in the future),
        if meta info not match, raise MetaInfoError
    Na: int
        output multipoles, default co.Nl
    Nl: int
        input multipoles, default co.Nl
    Nq: int
        use Nq Qs, default 3, i.e. use Q0, Q2, Q4
    pmax: float, default 0.3
        integrate window up to pmax
    accboost: int
        accboost x number of p grids, default 1
    withmask: bool
        set it True to mask window, default True
    windowk: float
        mask width, default 0.05, i.e. when k=0.1, only p in [0.1 - 0.05, 0.1 + 0.05] will be integrated
    Nmax: int
        FFTLog nodes, default 4096
    xmin_factor: float
        FFTLog xmin = smin * xmin_factor, where smin is determined by window_configspace_file
        default xmin_factor = 1.0
    xmax_factor: float
        FFTLog xmax = smax * xmax_factor, where smax is determined by window_configspace_file
        default xmax_factor = 100.
    bias: float
        FFTLog bias, default -1.6
    window_param: float | None
        FFTLog window parameter, default 1
    window_st: bool
        set it True to apply window effect to stochastic terms
    icc: IntegralConstraint | None
        integral constraint effect, default None
    name: str
        logger name, default 'pybird.window'
    snapshot: bool
        take snapshot after applying ``window``, by default False

    Notes
    -----
    ds in mask file should be small enough because the value will be multiplied by a spherical
    bessel function and then will be interpolated

    Math
    ----
    window convolution

    ..math:: P'_a(k) = \sum_{l}\int dp W(k, p)_{a, l}P_l(p)

    window

    ..math:: W(k, p)_{a, l} = \frac{2}{\pi}(-i)^a i^l p^2 \int ds s^2 j_{a}(ks)Q_{a,l}(s)j_l(ps)

    configspace window

    ..math:: Q_{a, l}(s) = \sum_{q}C_{a,l,q}Q_q(s)

    coef

    ..math:: C_{\ell, \ell^{\prime}, \ell^{\prime \prime}} \equiv(2 \ell+1)\left(\begin{array}{ccc}\ell & \ell^{\prime} & \ell^{\prime \prime} \\ 0 & 0 & 0\end{array}\right)^{2}
    """

    def __init__(
        self,
        window_fourier_file: Location | None = None,
        window_configspace_file: Location | None = None,
        co: Common = common,
        load: bool = True,
        save: bool = True,
        check_meta: bool = True,
        Na: int | None = None,
        Nl: int | None = None,
        Nq: int | None = 3,
        pmax: float | None = None,
        accboost: int = 1,
        withmask: bool = True,
        windowk: float = 0.05,
        Nmax: int = 4096,
        xmin_factor: float = 1.0,
        xmax_factor: float = 100.0,
        bias: float = -1.6,
        window_param: float | None = 1,
        window_st: bool = True,
        icc: IntegralConstraint | None = None,
        name: str = "pybird.window",
        snapshot: bool = False,
    ) -> None:
        self.set_logger(name=name)
        self.co: Common = co
        if window_fourier_file is None and window_configspace_file is None:
            raise ValueError(
                "Window requires window_fourier_file "
                "or window_configspace_file or both"
            )
        self.window_fourier_file = (
            Path(window_fourier_file).resolve() if window_fourier_file else None
        )
        self.window_configspace_file = (
            Path(window_configspace_file).resolve() if window_configspace_file else None
        )
        window_configspace_file_str = None
        if self.window_configspace_file:
            window_configspace_file_str = str(self.window_configspace_file)
        self._load = load
        self._save = save if self.window_fourier_file else False
        self.check_meta = check_meta
        self._create_meta = True
        self.window_st = window_st
        self.withmask = withmask
        self.windowk = windowk
        Na = Na if Na else self.co.Nl
        Nl = Nl if Nl else self.co.Nl
        if Na > self.co.Nl or Nl > self.co.Nl:
            raise ValueError(
                f"request Na={Na}, Nl={Nl} "
                f"while bird only compute Nl up to {self.co.Nl}"
            )
        if Na > Nl:
            raise ValueError(f"dangerous settings Na={Na}, Nl={Nl}")
        if pmax is None:
            pmax = float(self.co.k.max())
        self.p = window_kgrid(kmax=pmax, accboost=accboost)
        self.meta: dict[str, Any] = dict(
            Na=Na,
            Nl=Nl,
            Nq=Nq,
            pmax=pmax,
            accboost=accboost,
            Nmax=Nmax,
            xmin_factor=xmin_factor,
            xmax_factor=xmax_factor,
            bias=bias,
            window_param=window_param,
            window_configspace_file=window_configspace_file_str,
            k=self.co.k.tolist(),
        )
        self.Wal = self._load_Wal()  # type: ignore
        if self.Wal is None:
            self.Wal: NDArray = self._compute_Wal()
        self.Waldk: NDArray = self._compute_Waldk()  # mask settings not in meta
        if self._save:
            self._save_Wal()
        self.icc = icc
        self.snapshot = snapshot
        if self.snapshot:
            self.mpi_info("snapshot is enabled")

    def _load_Wal(self):
        """load fourier matrix, return None if failed"""
        Wal = None
        window_fourier_file = self.window_fourier_file
        if self._load:
            try:
                Wal = np.load(window_fourier_file)  # type: ignore
                self.mpi_info("Loaded mask: %s", window_fourier_file)
            except (OSError, TypeError):
                self.mpi_warning("Cannot load mask: %s", window_fourier_file)
            else:
                # retry: Backward Compatibility
                # automatically recompute if Nl not match
                if Wal.shape[1] != self.meta["Nl"]:
                    Wal = None
                    self.mpi_info(
                        "Nl not match, load the one with suffix '_Nl%d'",
                        self.meta["Nl"],
                    )
                    assert window_fourier_file
                    self.window_fourier_file = replace_suffix(
                        window_fourier_file, f'_Nl{self.meta["Nl"]}.npy'
                    )
                    window_fourier_file = self.window_fourier_file
                    try:
                        Wal = np.load(window_fourier_file)
                        self.mpi_info("Loaded mask: %s", window_fourier_file)
                    except OSError:
                        self.mpi_warning("Cannot load mask: %s", window_fourier_file)
            if (Wal is not None) and self.check_meta:
                assert window_fourier_file
                meta_file = replace_suffix(window_fourier_file, ".json")
                if not meta_file.exists():
                    self.mpi_warning(
                        "Deprecated: when check_meta=True and load=True, "
                        "Window will try to find the meta-info json file and "
                        "compare it to current settings. In the future, "
                        "if the meta-info file does not exist, "
                        "Exception will be raised."
                    )
                    self._create_meta = False
                else:
                    with meta_file.open("r") as f:
                        meta = json.load(f)
                    # copy meta info from previous file
                    if self.meta["window_configspace_file"] is None:
                        self.meta["window_configspace_file"] = meta[
                            "window_configspace_file"
                        ]
                    if meta != self.meta:
                        raise MetaInfoError(
                            f"inconsistent meta info\n"
                            f"loaded matrix's meta:\n{meta}\n"
                            f"expect:\n{self.meta}"
                        )
        if Wal is not None:
            self._save = False

        return Wal

    def _compute_Wal(self):
        self.mpi_info("Computing new mask")
        self.mpi_warning(
            "please manually check the fourier matrix after computation, "
            "since the default FFTLog settings may not be optimal"
        )
        Na, Nl, Nq = [cast(int, self.meta[x]) for x in ("Na", "Nl", "Nq")]
        if self.window_configspace_file is None:
            raise ValueError("please specify a configuration space mask file")

        try:
            swindow_config_space = np.loadtxt(self.window_configspace_file)
            # remove s=0
            while swindow_config_space[0, 0] == 0.0:
                swindow_config_space = swindow_config_space[1:, :]
            # keep Q0, Q2, ... (Nq) only
            if swindow_config_space.shape[-1] > (1 + Nq):
                swindow_config_space = swindow_config_space[:, : 1 + Nq]
        except OSError as ex:
            raise OSError(
                f"Error: can't load mask file: " f"{self.window_configspace_file}"
            )
        except IndexError as ex:
            raise TypeError(f"loaded config-space mask has unexpected shape") from ex

        # fmt: off
        Calq = np.array([
            [[     1,      0,        0,        0],
             [     0,    1/5,        0,        0],
             [     0,      0,      1/9,        0],
             [     0,      0,        0,     1/13]],
            [[     0,      1,        0,        0],
             [     1,    2/7,      2/7,        0],
             [     0,    2/7,  100/693,   25/143],
             [     0,      0,   25/143,   14/143]],
            [[     0,      0,        1,        0],
             [     0,  18/35,    20/77,   45/143],
             [     1,  20/77, 162/1001,   20/143],
             [     0, 45/143,   20/143, 252/2431]],
            [[     0,      0,        0,        1],
             [     0,      0,     5/11,    14/55],
             [     0,   5/11,    20/99,   28/187],
             [     1,  14/55,   28/187, 400/3553]],
        ])[..., :Nq]  # type: ignore
        # fmt: on

        sw = swindow_config_space[:, 0]
        Qq = swindow_config_space[:, 1:].T
        Qal = np.einsum("alq,qs->als", Calq, Qq)[:Na, :Nl, :]

        self.fftsettings = dict(
            Nmax=self.meta["Nmax"],
            xmin=sw[0] * self.meta["xmin_factor"],
            xmax=sw[-1] * self.meta["xmax_factor"],
            bias=self.meta["bias"],
        )  # 1e-2 - 1e6 [Mpc/h]
        self.fft = FFTLog(**self.fftsettings)
        self.pPow = np.exp(np.einsum("n,p->np", -self.fft.Pow - 3.0, np.log(self.p)))
        self.M = np.empty(shape=(Nl, self.fft.Pow.shape[0]), dtype="complex")
        for l in range(Nl):
            self.M[l] = 4 * np.pi * MPC(2 * l, -0.5 * self.fft.Pow)

        self.Coef = np.empty(
            shape=(Na, Nl, self.co.Nk, self.fft.Pow.shape[0]),
            dtype="complex",
        )
        Nas = np.arange(Na)
        Nls = np.arange(Nl)
        kernel = lambda x: spherical_jn(
            2 * Nas[:, newaxis, newaxis, newaxis],
            x[newaxis, newaxis, newaxis, :] * self.co.k[newaxis, newaxis, :, newaxis],
        )
        self.Coef[...] = (
            (-1j) ** (2 * Nas)[:, newaxis, newaxis, newaxis]
            * (1j) ** (2 * Nls)[newaxis, :, newaxis, newaxis]
            * self.fft.Coef(
                sw,
                Qal[:, :, newaxis, :]
                * np.ones(self.co.k.size)[newaxis, newaxis, :, newaxis],
                window=self.meta["window_param"],
                extrap="padding",
                kernel=kernel,
            )
        )

        Wal = self.p**2 * np.real(
            np.einsum("alkn,np,ln->alkp", self.Coef, self.pPow, self.M)
        )

        return Wal

    def _compute_Waldk(self):
        # Apply masking centered around the value of k
        Wal_masked = self.Wal
        if self.withmask:
            kpgrid, kgrid = np.meshgrid(self.p, self.co.k, indexing="ij")
            mask = (kpgrid < kgrid + self.windowk) & (kpgrid > kgrid - self.windowk)
            Wal_masked = np.einsum("alkp,pk->alkp", self.Wal, mask)

        # the spacing (need to do the convolution as a sum)
        deltap = self.p[1:] - self.p[:-1]  # type: ignore
        deltap = np.concatenate([[0], deltap])
        return np.einsum("alkp,p->alkp", Wal_masked, deltap)

    @root_only
    def _save_Wal(self):
        assert self.window_fourier_file
        self.mpi_info("Saving mask: %s", self.window_fourier_file)
        np.save(self.window_fourier_file, self.Wal)
        if self._create_meta:
            meta_file = replace_suffix(self.window_fourier_file, ".json")
            with meta_file.open("w") as f:
                json.dump(self.meta, f, indent=2)

    def integrWindow(self, P, interp=True):
        """
        Convolve the window functions to a power spectrum P
        """
        if interp:
            Pk = interp1d(
                self.co.k,
                P,
                axis=-1,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",  # type: ignore
            )(self.p)
        else:
            Pk = P
        # (multipole l, multipole ' p, k, k' m) , (multipole ', power pectra s, k' m)
        return np.einsum("alkp,lsp->ask", self.Waldk, Pk, optimize=True)

    def Window(self, bird: Bird):
        """
        Apply the survey window function to the bird power spectrum
        """
        if self.icc:
            icc = self.icc
            bird.P11l = self.integrWindow(bird.P11l) - icc.integrWindow(bird.P11l)
            bird.Pctl = self.integrWindow(bird.Pctl) - icc.integrWindow(bird.Pctl)
            bird.Ploopl = self.integrWindow(bird.Ploopl) - icc.integrWindow(bird.Ploopl)
            # fmt: off
            if bird.co.with_NNLO:
                bird.PctNNLOl = self.integrWindow(bird.PctNNLOl) - icc.integrWindow(bird.PctNNLOl)
            # fmt: on
            if self.window_st:
                # TODO: icc_st?
                bird.Pstl = self.integrWindow(bird.Pstl) - icc.integrWindow(bird.Pstl)
            bird.Picc -= icc.PSN  # type: ignore
        else:
            bird.P11l = self.integrWindow(bird.P11l)
            bird.Pctl = self.integrWindow(bird.Pctl)
            bird.Ploopl = self.integrWindow(bird.Ploopl)
            if bird.co.with_NNLO:
                bird.PctNNLOl = self.integrWindow(bird.PctNNLOl)
            if self.window_st:
                bird.Pstl = self.integrWindow(bird.Pstl)
        if self.snapshot:
            bird.create_snapshot("window")


@dataclass
class PInfo:
    ells: tuple[int, ...]
    kmin: float
    kmax: float
    nbins: int


def to_window_matrix(
    matrix,
    inpoles: PInfo,
    outpoles: PInfo,
    ells_in: tuple[int, ...],
    kmax_in: float,
    ells_out: tuple[int, ...],
    kmin_out: float,
    kmax_out: float,
):
    # XXX: didn't account for the order of ells
    # step1: mask input
    kedges = np.linspace(inpoles.kmin, inpoles.kmax, inpoles.nbins + 1)
    kin = (kedges[1:] + kedges[:-1]) / 2
    mask_in = np.zeros(inpoles.nbins * len(inpoles.ells), dtype=bool)
    ileft, iright = 0, np.searchsorted(kin, kmax_in)
    for ell in inpoles.ells:
        if ell in ells_in:
            mask_in[ileft:iright] = True
        ileft, iright = ileft + inpoles.nbins, iright + inpoles.nbins

    # step2: mask output
    kedges = np.linspace(outpoles.kmin, outpoles.kmax, outpoles.nbins + 1)
    kout = (kedges[1:] + kedges[:-1]) / 2
    mask_out = np.zeros(outpoles.nbins * len(outpoles.ells), dtype=bool)
    ileft, iright = np.searchsorted(kout, kmin_out), np.searchsorted(kout, kmax_out)
    for ell in outpoles.ells:
        if ell in ells_out:
            mask_out[ileft:iright] = True
        ileft, iright = ileft + outpoles.nbins, iright + outpoles.nbins

    # step3: rearrange the shape
    matrix = matrix[np.ix_(mask_out, mask_in)]
    nk_out = matrix.shape[0] // len(ells_out)
    nk_in = matrix.shape[1] // len(ells_in)
    retval = np.zeros((len(ells_out), len(ells_in), nk_out, nk_in))
    for i, _ in enumerate(ells_out):
        for j, _ in enumerate(ells_in):
            idx = np.ix_(
                np.arange(i * nk_out, (i + 1) * nk_out),
                np.arange(j * nk_in, (j + 1) * nk_in),
            )
            retval[i, j] = matrix[idx]
    return retval


class PolesInfo(NamedTuple):
    nells: int
    kstart: float
    kend: float
    nbin: int


@dataclass
class WindowMatrix(HasLogger):
    matrix: ndarrayf
    inpoles: PolesInfo
    outpoles: PolesInfo
    co: Common = field(default_factory=Common)
    window_st: bool = False
    icc: IntegralConstraint | None = None
    name: str = "pybird.WindowMatrix"
    snapshot: bool = False

    def __post_init__(self):
        self.set_logger(name=self.name)
        if self.icc:
            raise NotImplementedError("ICC not implemented for WindowMatrix")
        if self.matrix.shape != (
            self.outpoles.nells,
            self.inpoles.nells,
            self.outpoles.nbin,
            self.inpoles.nbin,
        ):
            print(self.matrix.shape)
            print(
                self.outpoles.nells,
                self.inpoles.nells,
                self.outpoles.nbin,
                self.inpoles.nbin,
            )
            raise ValueError("matrix shape does not match meta information")
        if self.inpoles.nells != self.co.Nl:
            raise ValueError("input poles do not match self.co.Nl")

    @classmethod
    def load(
        cls,
        path: str,
        ells: list[int],
        kmin: float,
        kmax: float,
        co: Common = Common(),
        window_st: bool = False,
        icc: IntegralConstraint | None = None,
        name: str = "pybird.WindowMatrix",
        snapshot: bool = False,
    ):
        # hard-coded to match Florian Window
        matrix = np.loadtxt(path)
        matrix = to_window_matrix(
            matrix,
            PInfo(ells=(0, 2, 4), kmin=0, kmax=0.4, nbins=400),
            PInfo(ells=(0, 1, 2, 3, 4), kmin=0, kmax=0.4, nbins=40),
            ells_in=tuple(2 * _ for _ in range(co.Nl)),
            kmax_in=co.k.max(),
            ells_out=tuple(ells),
            kmin_out=kmin,
            kmax_out=kmax,
        )
        return WindowMatrix(
            matrix,
            inpoles=PolesInfo(co.Nl, 0, co.k.max(), matrix.shape[3]),
            outpoles=PolesInfo(len(ells), kmin, kmax, matrix.shape[2]),
            co=co,
            window_st=window_st,
            icc=icc,
            name=name,
            snapshot=snapshot,
        )

    @cached_property
    def kavg(self):
        # hard-coded for test
        return np.linspace(0, 0.4, 400)[:300]
        kedges = np.linspace(
            self.inpoles.kstart, self.inpoles.kend, self.inpoles.nbin + 1
        )
        khigh = kedges[1:]
        klow = kedges[:-1]
        return (1 / 4 * (khigh**4 - klow**4)) / (1 / 3 * (khigh**3 - klow**3))

    def convolve(self, Plk):
        # fmt: off
        Plk = interp1d(
            self.co.k, Plk,
            axis=-1, kind="cubic", bounds_error=False, fill_value="extrapolate",
        )(self.kavg)
        # fmt: on
        return np.einsum("alkp,l...p->a...k", self.matrix, Plk, optimize=True)

    def Window(self, bird: Bird):
        bird.P11l = self.convolve(bird.P11l)
        bird.Pctl = self.convolve(bird.Pctl)
        bird.Ploopl = self.convolve(bird.Ploopl)
        if bird.co.with_NNLO:
            bird.PctNNLOl = self.convolve(bird.PctNNLOl)
        if self.window_st:
            bird.Pstl = self.convolve(bird.Pstl)
        # still include Picc...
        bird.Picc = self.convolve(bird.Picc)
        bird.Picc[...] = 0.0
        if self.snapshot:
            bird.create_snapshot("window")
