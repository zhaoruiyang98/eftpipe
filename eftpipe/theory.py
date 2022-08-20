from __future__ import annotations

# global
import itertools
import time
import numpy as np
from collections import defaultdict
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Any, cast, Iterable, NamedTuple, TypedDict, TYPE_CHECKING
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import legendre
from cobaya.log import HasLogger
from cobaya.log import LoggedError
from cobaya.theory import HelperTheory
from cobaya.theory import Theory

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from cobaya.theory import Provider

# local
from .interface import CobayaCambInterface
from .interface import CobayaClassyInterface
from .pybird import pybird
from .tools import bool_or_list
from .tools import group_lists
from .tools import int_or_list
from .tools import Initializer
from .tools import recursively_update_dict


def chain_coeff(l: int) -> float:
    r"""compute ``A_\ell`` coeff for chained power spectrum

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


class BirdPlus(pybird.Bird):
    """enhanced version of pybird.Bird, support multi-tracer, hook and snapshot
    """

    _hooks: list[BirdHook]
    b1A: float
    b1B: float
    b11AB: NDArray[np.float64]
    bctAB: NDArray[np.float64]
    bloopAB: NDArray[np.float64]
    bstAB: NDArray[np.float64]
    fullPs: NDArray[np.float64]
    PG: dict[str, NDArray[np.float64]]

    # override
    def initialize(self) -> None:
        self._hooks: list[BirdHook] = []
        self.snapshots: dict[str, BirdSnapshot] = {}

    def attach_hook(self, *args: BirdHook) -> None:
        self._hooks.extend(args)

    def clear_hook(self) -> None:
        self._hooks = []

    def create_snapshot(self, name: str) -> None:
        """
        Notes
        -----
        be careful with the name, because name confliction not checked
        """
        # XXX: name confliction not checked
        snapshot = BirdSnapshot(self)
        self.snapshots[name] = snapshot
        self.attach_hook(snapshot)

    # override
    def setreducePslb(
        self,
        bsA: Iterable[float],
        bsB: Iterable[float] | None = None,
        es: Iterable[float] = (0.0, 0.0, 0.0),
    ) -> None:
        """apply counter terms and bind fullPs to self

        Parameters
        ----------
        bsA : Iterable[float]
            b_1, b_2, b_3, b_4, c_{ct}, c_{r,1}, c{r,2}
        bsB : Iterable[float], optional
            the same as bsA, but for tracer B, by default None,
            and will compute auto power spectrum
        es : Iterable[float], optional
            c_{e,0}, c_{mono}, c_{quad}, by default zeros
        """
        kmA, ndA, kmB, ndB = self.co.kmA, self.co.ndA, self.co.kmB, self.co.ndB
        b1A, b2A, b3A, b4A, cctA, cr1A, cr2A = bsA
        if bsB is None:
            bsB = bsA
        b1B, b2B, b3B, b4B, cctB, cr1B, cr2B = bsB
        f = self.f
        ce0, cemono, cequad = es

        # cct -> cct / km**2, cr1 -> cr1 / km**2, cr2 -> cr2 / km**2
        # ce0 -> ce0 / nd, cemono -> cemono / nd / km**2, cequad -> cequad / nd / km**2
        b11AB = np.array([b1A * b1B, (b1A + b1B) * f, f ** 2])
        bctAB = np.array(
            [
                b1A * cctB / kmB ** 2 + b1B * cctA / kmA ** 2,
                b1B * cr1A / kmA ** 2 + b1A * cr1B / kmB ** 2,
                b1B * cr2A / kmA ** 2 + b1A * cr2B / kmB ** 2,
                (cctA / kmA ** 2 + cctB / kmB ** 2) * f,
                (cr1A / kmA ** 2 + cr1B / kmB ** 2) * f,
                (cr2A / kmA ** 2 + cr2B / kmB ** 2) * f,
            ]
        )
        bloopAB = np.array(
            [
                1.0,
                1.0 / 2.0 * (b1A + b1B),
                1.0 / 2.0 * (b2A + b2B),
                1.0 / 2.0 * (b3A + b3B),
                1.0 / 2.0 * (b4A + b4B),
                b1A * b1B,
                1.0 / 2.0 * (b1A * b2B + b1B * b2A),
                1.0 / 2.0 * (b1A * b3B + b1B * b3A),
                1.0 / 2.0 * (b1A * b4B + b1B * b4A),
                b2A * b2B,
                1.0 / 2.0 * (b2A * b4B + b2B * b4A),
                b4A * b4B,
            ]
        )
        xfactor1 = 0.5 * (1.0 / ndA + 1.0 / ndB)
        xfactor2 = 0.5 * (1.0 / ndA / kmA ** 2 + 1.0 / ndB / kmB ** 2)
        bstAB = np.array([ce0 * xfactor1, cemono * xfactor2, cequad * xfactor2])

        self.b11AB = b11AB
        self.bctAB = bctAB
        self.bloopAB = bloopAB
        self.bstAB = bstAB
        self.fullPs = self.reducePslb(
            b11AB=b11AB,
            bloopAB=bloopAB,
            bctAB=bctAB,
            bstAB=bstAB,
            P11l=self.P11l,
            Ploopl=self.Ploopl,
            Pctl=self.Pctl,
            Pstl=self.Pstl,
        )
        for viewer in self._hooks:
            viewer.setreducePslb(self)

    def reducePslb(
        self, *, b11AB, bloopAB, bctAB, bstAB, P11l, Ploopl, Pctl, Pstl
    ) -> NDArray:
        Ps0 = np.einsum("b,lbx->lx", b11AB, P11l)
        Ps1 = np.einsum("b,lbx->lx", bloopAB, Ploopl) + np.einsum(
            "b,lbx->lx", bctAB, Pctl
        )
        Ps2 = np.einsum("b,lbx->lx", bstAB, Pstl)
        return Ps0 + Ps1 + Ps2

    def setreducePG(self, b1A: float, b1B: float) -> None:

        self.b1A = b1A
        self.b1B = b1B

        self.PG = self.reducePG(
            b1A=b1A, b1B=b1B, Ploopl=self.Ploopl, Pctl=self.Pctl, Pstl=self.Pstl
        )

        for viewer in self._hooks:
            viewer.setreducePG(self)

    def reducePG(
        self, b1A: float, b1B: float, Ploopl: NDArray, Pctl: NDArray, Pstl: NDArray
    ) -> dict[str, NDArray]:
        f = self.f
        kmA, ndA, kmB, ndB = self.co.kmA, self.co.ndA, self.co.kmB, self.co.ndB
        PG: dict[str, Any] = {}
        PG["b3A"] = 1 / 2 * Ploopl[:, 3, :] + 1 / 2 * b1B * Ploopl[:, 7, :]
        PG["cctA"] = b1B / kmA ** 2 * Pctl[:, 0, :] + f / kmA ** 2 * Pctl[:, 3, :]
        PG["cr1A"] = b1B / kmA ** 2 * Pctl[:, 1, :] + f / kmA ** 2 * Pctl[:, 4, :]
        PG["cr2A"] = b1B / kmA ** 2 * Pctl[:, 2, :] + f / kmA ** 2 * Pctl[:, 5, :]
        PG["b3B"] = 1 / 2 * Ploopl[:, 3, :] + 1 / 2 * b1A * Ploopl[:, 7, :]
        PG["cctB"] = b1A / kmB ** 2 * Pctl[:, 0, :] + f / kmB ** 2 * Pctl[:, 3, :]
        PG["cr1B"] = b1A / kmB ** 2 * Pctl[:, 1, :] + f / kmB ** 2 * Pctl[:, 4, :]
        PG["cr2B"] = b1A / kmB ** 2 * Pctl[:, 2, :] + f / kmB ** 2 * Pctl[:, 5, :]
        xfactor1 = 0.5 * (1.0 / ndA + 1.0 / ndB)
        xfactor2 = 0.5 * (1.0 / ndA / kmA ** 2 + 1.0 / ndB / kmB ** 2)
        PG["ce0"] = Pstl[:, 0, :] * xfactor1
        PG["cemono"] = Pstl[:, 1, :] * xfactor2
        PG["cequad"] = Pstl[:, 2, :] * xfactor2
        return PG


class BirdHook:
    """accept a BirdPlus object, and then do postprocessing
    """

    def setreducePslb(self, bird: BirdPlus) -> None:
        """automatically invoked when BirdPlus.setreducePslb is called
        """
        pass

    def setreducePG(self, bird: BirdPlus) -> None:
        """automatically invoked when BirdPlus.setreducePG is called
        """
        pass


class BirdSnapshot(BirdHook):
    """
    created by BridPlus.create_snapshot, do not use this class directly
    """

    def __init__(self, bird: BirdPlus) -> None:
        self.k = bird.co.k.copy()
        self.ls = [2 * i for i in range(bird.co.Nl)]
        self.P11l = bird.P11l.copy()
        self.Ploopl = bird.Ploopl.copy()
        self.Pctl = bird.Pctl.copy()
        self.Pstl = bird.Pstl.copy()

    def setreducePslb(self, bird: BirdPlus) -> None:
        self.fullPs = bird.reducePslb(
            b11AB=bird.b11AB,
            bloopAB=bird.bloopAB,
            bctAB=bird.bctAB,
            bstAB=bird.bstAB,
            P11l=self.P11l,
            Ploopl=self.Ploopl,
            Pctl=self.Pctl,
            Pstl=self.Pstl,
        )


class Binning(BirdHook, HasLogger):
    """Match the theoretical output to data, doing binning

    Parameters
    ----------
    kout: ArrayLike, 1d
        k of data
    accboost: int
        accuracy boost, default 1, do integration using 100 points per bin
    decimals: int
        compute delta_k by rounding the difference of last two kout.
        Default is 2 and this works well when delta_k = 0.01
    co: Common
        this class only uses co.k, default pybird.Common
    name: str
        logger name, by default 'pybird.binning'

    Methods
    -------
    kbinning(bird: BirdPlus): apply binning

    Notes
    -----
    kbins will be constructed using kout[-1] - kout[-2]
    """

    def __init__(
        self,
        kout,
        accboost: int = 1,
        decimals: int = 2,
        co: pybird.Common = pybird.common,
        name: str = "pybird.binning",
    ) -> None:
        self.set_logger(name=name)
        self.kout = np.array(kout)
        self.co = co
        self.accboost = accboost
        self.decimals = decimals
        self.mpi_info("binning correction: on")
        self.mpi_info(
            "%d data points, from %.3f to %.3f",
            self.kout.size,
            self.kout[0],
            self.kout[-1],
        )
        kspaces = np.around(self.kout[1:] - self.kout[:-1], decimals=decimals)  # type: ignore
        kspace_diff = kspaces[1:] - kspaces[:-1]
        if not np.allclose(kspace_diff, 0, rtol=0, atol=1e-6):
            self.mpi_warning(
                "binning correction on, "
                "but given kout seems not linearly spaced, "
                "be careful because the constructed kbins may be wrong, "
                "especially when 'kmax' is small",
            )
        self.loadBinning(self.kout)
        self.mpi_info("num of kgrids in each bin: %d", self.points[0].size)
        self.mpi_info(
            "round the difference of last two kout to %d decimal places", self.decimals,
        )

    def loadBinning(self, setkout) -> None:
        """
        Create the bins of the data k's
        """
        delta_k = np.round(setkout[-1] - setkout[-2], self.decimals)
        kcentral = (setkout[-1] - delta_k * np.arange(len(setkout)))[::-1]
        binmin = kcentral - delta_k / 2
        binmax = kcentral + delta_k / 2
        self.binvol = np.array(
            [
                quad(lambda k: k ** 2, kbinmin, kbinmax)[0]
                for (kbinmin, kbinmax) in zip(binmin, binmax)
            ]
        )
        self.keff = np.array(
            [
                quad(lambda k: k * k ** 2, kbinmin, kbinmax)[0]
                for (kbinmin, kbinmax) in zip(binmin, binmax)
            ]
        )
        self.keff = self.keff / self.binvol
        points = [
            np.linspace(kbinmin, kbinmax, 100 * self.accboost)
            for (kbinmin, kbinmax) in zip(binmin, binmax)
        ]
        self.points = np.array(points)

    def integrBinning(self, P: NDArray) -> NDArray:
        """
        Integrate over each bin of the data k's
        """
        Pkint = interp1d(
            self.co.k,
            P,
            axis=-1,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        res = np.trapz(Pkint(self.points) * self.points ** 2, x=self.points, axis=-1)
        return res / self.binvol

    def kbinning(self, bird: BirdPlus) -> None:
        """
        Apply binning in k-space for linear-spaced data k-array
        """
        self.P11l = self.integrBinning(bird.P11l)
        self.Pctl = self.integrBinning(bird.Pctl)
        self.Ploopl = self.integrBinning(bird.Ploopl)
        self.Pstl = self.integrBinning(bird.Pstl)

    # override
    def setreducePslb(self, bird: BirdPlus) -> None:
        self.fullPs = bird.reducePslb(
            b11AB=bird.b11AB,
            bloopAB=bird.bloopAB,
            bctAB=bird.bctAB,
            bstAB=bird.bstAB,
            P11l=self.P11l,
            Ploopl=self.Ploopl,
            Pctl=self.Pctl,
            Pstl=self.Pstl,
        )

    # override
    def setreducePG(self, bird: BirdPlus) -> None:
        self.PG = bird.reducePG(
            b1A=bird.b1A,
            b1B=bird.b1B,
            Ploopl=self.Ploopl,
            Pctl=self.Pctl,
            Pstl=self.Pstl,
        )


class EFTLSS(Theory):
    """
    Effective field theory of Large-scale Structures
    """

    file_base_name = "eftlss"

    cache_dir_path: Path
    tracers: dict[str, dict[str, Any]]

    names: list[str]

    def initialize(self) -> None:
        super().initialize()
        self.cache_dir_path = Path(self.cache_dir_path)
        # XXX: self.tracers seems not completely copied by cobaya
        self.tracers = deepcopy(self.tracers)
        if "common" in self.tracers.keys():
            self.mpi_info("'common' field is used as default settings")
            common = self.tracers.pop("common")
            for k, tracer_config in self.tracers.items():
                common_config = deepcopy(common)
                recursively_update_dict(common_config, tracer_config)
                self.tracers[k] = common_config
        self.names = list(self.tracers.keys())
        if not self.names:
            raise LoggedError(self.log, "No tracer specified")

        # check cross
        for name, tracer_config in self.tracers.items():
            cross = tracer_config.get("cross", [])
            if cross and (not isinstance(cross, list) or len(cross) != 2):
                raise LoggedError(
                    self.log,
                    "tracer %s: expect a list of 2 elements, but given cross=%r",
                    name,
                    cross,
                )
            if diff := set(cross).difference(self.names):
                raise LoggedError(
                    self.log,
                    "tracer %s: cross=%r contains unknown tracer names: %r",
                    name,
                    cross,
                    diff,
                )

        # check boltzmann provider
        providers = set(v.get("provider", "classy") for v in self.tracers.values())
        if len(providers) != 1:
            raise LoggedError(self.log, "All tracers should share the same provider")
        supported_providers = ["camb", "classy", "classynu"]
        if providers.pop() not in supported_providers:
            raise LoggedError(self.log, "supported providers: %s", supported_providers)

        # {requirement: tracers' names}
        self._must_provide: defaultdict[str, set[str]] = defaultdict(set)

    def get_requirements(self) -> dict[str, dict]:
        """
        make it possible to use eftlss with likelihood one, otherwise eftlss may not depend on anything
        """
        return {name + "_results": {} for name in self.names}

    def must_provide(
        self, **requirements: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """redirect requirements to EFTLSSChild

        For example, ``nonlinear_Plk_grid={{'LRG': {...}, 'ELG': {...}}}`` will 
        be packed like ``{'LRG_results': {'nonlinear_Plk_grid': {...}}, 'ELG_results': {'nonlinear_Plk_grid': {...}}}``

        all requirements support ``common`` field

        detailed requirement form can be found in ``EFTLSSChild.must_provide``
        """
        super().must_provide(**requirements)
        redirected_reqs_tmp: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        for product, config_dict in requirements.items():
            self._must_provide[product] |= config_dict.keys()
            if product == "snapshots":
                # no settings required for snapshots
                continue
            for tracer, settings in config_dict.items():
                redirected_reqs_tmp[tracer][product] = settings
        redirected_reqs = {}
        for tracer, config_dict in redirected_reqs_tmp.items():
            redirected_reqs[tracer + "_results"] = deepcopy(config_dict)
        if common := redirected_reqs_tmp.pop("common", {}):
            for tracer, config_dict in redirected_reqs.items():
                ref = deepcopy(common)
                recursively_update_dict(ref, config_dict)
                redirected_reqs[tracer] = ref
        return redirected_reqs

    def calculate(self, state, want_derived=True, **params_values_dict):
        pass

    def _get_tracer_products(self, tracer: str, product: str):
        if tracer not in self._must_provide[product]:
            raise LoggedError(self.log, "No %s requested for %s", product, tracer)
        return self.provider.get_result(tracer + "_results")[product]

    def get_nonlinear_Plk_grid(
        self, tracer: str, chained: bool = False, binned: bool = False
    ) -> tuple[list[int], NDArray, NDArray]:
        results = self._get_tracer_products(tracer, "nonlinear_Plk_grid")
        key = PlkKey(chained=chained, binned=binned)
        # let it crash...
        return results[key]

    def get_nonlinear_Plk_gaussian_grid(
        self, tracer: str, chained: bool = False, binned: bool = False
    ) -> tuple[list[int], NDArray, dict[str, NDArray]]:
        results = self._get_tracer_products(tracer, "nonlinear_Plk_gaussian_grid")
        key = PlkKey(chained=chained, binned=binned)
        return results[key]

    def get_nonlinear_Plk_interpolator(
        self, tracer: str, chained: bool = False
    ) -> PlkInterpolator:
        results = self._get_tracer_products(tracer, "nonlinear_Plk_interpolator")
        key = PlkKey(chained=chained, binned=False)
        return results[key]

    def get_snapshots(self, tracer: str) -> dict[str, BirdSnapshot]:
        return self._get_tracer_products(tracer, "snapshots")

    def get_helper_theories(self) -> dict[str, Theory]:
        out = {}
        for name in self.names:
            out[name] = EFTLSSChild(
                self, name, dict(stop_at_error=self.stop_at_error), timing=self.timer
            )
        # Pk_interpolator requires at least 4 redshift
        if len(out) < 4:
            first: EFTLSSChild = out[self.names[0]]
            first._zextra = [first.zeff + i * 0.1 for i in range(1, 5 - len(out))]
        return out


class PluginsDict(TypedDict):
    IRresum: pybird.Resum
    APeffect: pybird.APeffect
    window: pybird.Window
    fiber: pybird.FiberCollision


class PlkKey(NamedTuple):
    chained: bool
    binned: bool


def to_chained(ls_tot: list[int], fullPs: NDArray) -> NDArray:
    out = np.empty_like(fullPs)
    for i, l in enumerate(ls_tot[:-1]):
        out[i, :] = fullPs[i, :] - chain_coeff(l) * fullPs[i + 1, :]
    return out[:-1, :]


class PlkInterpolator:
    def __init__(self, ls: list[int], kgrid: NDArray, Plk: NDArray):
        self.ls = ls.copy()
        # XXX: not sure if inserting zero is a good idea
        kgrid = np.hstack(([0], kgrid))
        Plk = np.insert(Plk, 0, 0, axis=-1)
        tmp = interp1d(
            kgrid,
            kgrid * Plk,
            axis=-1,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        fn = lambda k: tmp(k) / k
        self.fn = fn

    def __call__(self, l: int | Iterable[int], k: ArrayLike) -> NDArray:
        l = int_or_list(l)
        try:
            idx = [self.ls.index(ll) for ll in l]
        except ValueError as ex:
            raise ValueError(f"l={l} not in {self.ls}") from ex
        return self.fn(k)[idx]


class EFTLSSChild(HelperTheory):
    def __init__(self, eftlss: EFTLSS, name: str, info, timing=None) -> None:
        # append extra redshifts if using classy (classy's bug)
        self._zextra: list[float] = []
        self.name = name
        self.eftlss = eftlss
        # EFTLSSChild always initialized after EFTLSS, safe to use config
        self.config = self.eftlss.tracers[name]
        super().__init__(info, self.eftlss.get_name() + "." + name, timing=timing)

    def initialize(self) -> None:
        super().initialize()
        self._not_reported = defaultdict(lambda: True)
        # allow empty string
        prefix = self.config.get("prefix", None)
        if prefix is None:
            prefix = self.name + "_"
        self.prefix: str = prefix
        # write back
        self.config["prefix"] = prefix
        try:
            self.zeff: float = self.config["z"]
        except KeyError:
            raise LoggedError(self.log, "must specify effective redshift z")
        # provider_name, use_cb and optiresum should be determined at initialization
        self.provider_name: str = self.config.get("provider", "classy")
        self.use_cb: bool = self.config.get("use_cb", False)
        self.optiresum: bool = self.config.get("IRresum", {}).pop("optiresum", False)
        # delayed initialization in initialize_with_provider,
        # so that all dependencies can be quickly checked
        # check config
        self.plugins: dict[str, Any] = {}
        self.with_IRresum: bool = self.config.get("with_IRresum", True)
        if self.with_IRresum:
            self.plugins["_IRresum"] = Initializer(
                pybird.Resum, self.config.get("IRresum", {}), self.log
            )
        self.with_APeffect: bool = self.config.get("with_APeffect", False)
        if self.with_APeffect:
            self.plugins["_APeffect"] = Initializer(
                pybird.APeffect, self.config.get("APeffect", {}), self.log
            )
        self.with_window: bool = self.config.get("with_window", False)
        if self.with_window:
            self.plugins["_window"] = Initializer(
                pybird.Window, self.config.get("window", {}), self.log
            )
        self.with_fiber: bool = self.config.get("with_fiber", False)
        if self.with_window:
            self.plugins["_fiber"] = Initializer(
                pybird.FiberCollision, self.config.get("fiber", {}), self.log
            )
        self.binning: Binning | None = None
        self._must_provide: defaultdict[str, dict[str, Any]] = defaultdict(dict)

    def initialize_with_provider(self, provider: Provider):
        super().initialize_with_provider(provider)

        use_cb = self.use_cb
        if self.provider_name == "camb":
            self.boltzmann = CobayaCambInterface(
                provider=provider, z=self.zeff, use_cb=use_cb
            )
        elif self.provider_name in ("classy", "classynu"):
            self.boltzmann = CobayaClassyInterface(
                provider=provider, z=self.zeff, use_cb=use_cb
            )
        else:
            raise LoggedError(
                self.log, "This should not happen, please contact the authors"
            )

        cross = self.config.get("cross", [])
        if not cross:
            try:
                kmA, ndA = self.config["km"], self.config["nd"]
                kmB, ndB = kmA, ndA
            except KeyError:
                raise LoggedError(self.log, "must specify km and nd")
        else:
            try:
                tracerA, tracerB = (self.eftlss.tracers[_] for _ in cross)
                kmA, ndA = tracerA["km"], tracerA["nd"]
                kmB, ndB = tracerB["km"], tracerB["nd"]
            except KeyError:
                raise LoggedError(self.log, "missing km or nd for tracer %s", cross)

        # do not initialize if no power spectrum requested
        if not self.need_power:
            return
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.mpi_info("effective redshift: %.3f", self.zeff)
        if cross:
            self.mpi_info(
                "computing cross power spectrum between %s and %s", cross[0], cross[1]
            )
        # TODO: set Nl according to plugins, e.g. window
        ls_group = []
        for name in (
            "nonlinear_Plk_grid",
            "nonlinear_Plk_interpolator",
            "nonlinear_Plk_gaussian_grid",
        ):
            if name not in self._must_provide:
                continue
            ls_group.append(self._must_provide[name]["ls"])
        # XXX: pybird actually does not support computing, e.g. P4 only
        lmax = max(group_lists(*ls_group))
        ls = list(range(0, lmax + 2, 2))
        Nl = len(ls)
        self.mpi_info("compute power spectrum multipoles (internal): %s", ls)

        self.co = pybird.Common(
            Nl=Nl, optiresum=self.optiresum, kmA=kmA, ndA=ndA, kmB=kmB, ndB=ndB
        )
        self.bird: BirdPlus | None = None
        self.nonlinear = pybird.NonLinear(
            load=True, save=True, co=self.co, path=str(self.eftlss.cache_dir_path)
        )

        if use_cb:
            self.mpi_info("using P_cb as input linear power spectrum")
        else:
            self.mpi_info("using P_m as input linear power spectrum")

        msg_pool = []
        if self.with_IRresum:
            self.plugins["IRresum"] = self.plugins["_IRresum"].initialize(co=self.co)
            msg_pool.append(
                ("IRresum enabled: %s", "optimized" if self.optiresum else "full")
            )
        if self.with_APeffect:
            self.plugins["APeffect"] = self.plugins["_APeffect"].initialize(
                co=self.co, name=self.get_name() + ".APeffect"
            )
            msg_pool.append(("APeffect enabled",))
        if self.with_window:
            self.plugins["window"] = self.plugins["_window"].initialize(
                co=self.co, name=self.get_name() + ".window"
            )
            msg_pool.append(("window enabled",))
        if self.with_fiber:
            self.plugins["fiber"] = self.plugins["_fiber"].initialize(
                co=self.co, name=self.get_name() + ".fiber"
            )
            msg_pool.append(("fiber enabled",))
        for msg in msg_pool:
            self.mpi_info(*msg)

        if self.need_binning:
            self.binning = Binning(
                **self._must_provide["nonlinear_Plk_grid"]["binning"],
                co=self.co,
                name=self.get_name() + ".binning",
            )

    def get_requirements(self) -> dict[str, Any]:
        # requirements should be known after initialization especially EFT parameters
        # XXX: dependency on cosmology may be dynamically updated
        z = self.zeff
        extra_zs = [] if z == 0.0 else [0.0]
        vars_pairs = [2 * ["delta_nonu"]] if self.use_cb else [2 * ["delta_tot"]]
        if self.provider_name == "camb":
            requires = {
                "Pk_interpolator": {
                    "nonlinear": False,
                    "z": [z] + self._zextra,
                    "k_max": 5,
                    "vars_pairs": vars_pairs,
                },
                "Hubble": {"z": extra_zs + [z]},
                "angular_diameter_distance": {"z": [z]},
                "fsigma8": {"z": [z]},
                "sigma8_z": {"z": [z]},
                "rdrag": None,
            }
        else:
            # sometimes classy does not reach the requested maximum redshift
            try:
                import classy

                if classy.__version__ == "v3.2.0":  # type: ignore
                    extra_zs += [self.zeff + 0.5]
            except ImportError:
                extra_zs += [self.zeff + 0.5]
            requires = {
                "Pk_interpolator": {
                    "nonlinear": False,
                    "z": [z] + extra_zs,
                    "k_max": 5,
                    "vars_pairs": vars_pairs,
                },
                "Hubble": {"z": extra_zs + [z]},
                "angular_diameter_distance": {"z": [z]},
                "fsigma8": {"z": [z]},
                "rdrag": None,
            }

        # TODO: some parameters can be moved to get_can_support_params
        # the following lines are reused by ``_params_reader``, possible to combine?
        stnames = ("ce0", "cemono", "cequad")
        names = ("b1", "b2", "b3", "b4", "cct", "cr1", "cr2")
        cross = self.config.get("cross", [])
        if cross:
            eft_params = [self.prefix + name for name in stnames]
            cross_prefix = [self.eftlss.tracers[_]["prefix"] for _ in cross]
            for prefix in cross_prefix:
                eft_params += [prefix + name for name in names]
        else:
            eft_params = [self.prefix + name for name in names]
            eft_params += [self.prefix + name for name in stnames]
        eft_requires = {param: None for param in eft_params}
        requires.update(eft_requires)
        return requires

    def _config_ls(
        self,
        k: str,
        v: dict[str, Any],
        chained: list[bool] = [False],
        base: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        k : str
            key name
        v : dict[str, Any]
            requires
        chained : list[bool]
            chained or not, used for ls reinterpretation
        base : str | None
            update associated base

        Notes
        -----
        saved ls always denote the internal ls and are always sorted in ascending order
        """
        stored = self._must_provide[k]
        ls: list[int] = stored.get("ls", [])
        try:
            ls_get = int_or_list(v.pop("ls"))
        except KeyError:
            raise LoggedError(self.log, "missing ls")
        # reinterpret ls if chained
        if True in chained:
            if len(chained) == 1:
                # e.g. [2] -> [0, 2, 4]
                tmp = [2 * _ for _ in range(min(ls_get) // 2 + 1)]
                ls_get = group_lists(tmp, [_ + 2 for _ in ls_get])
            else:
                # e.g. [2, 4] -> [0, 2, 4]
                tmp = [2 * _ for _ in range(min(ls_get) // 2)]
                ls_get = group_lists(tmp, ls_get)
        for l in ls_get:
            if l not in (0, 2, 4):
                raise LoggedError(
                    self.log,
                    "internal ls should be 0, 2 or 4, but requires %s",
                    ls_get,
                )
        combined_ls = group_lists(ls, ls_get)
        stored["ls"] = combined_ls
        if base:
            base_stored = self._must_provide[base]
            ls: list[int] = base_stored.get("ls", [])
            combined_ls = group_lists(ls, combined_ls)
            base_stored["ls"] = combined_ls

    def _config_binned(self, k: str, v: dict[str, Any]) -> None:
        stored = self._must_provide[k]
        binned: list[bool] = stored.get("binned", [])
        binning: dict[str, Any] = stored.setdefault("binning", {})  # default {}
        binned_get = bool_or_list(v.pop("binned", False))  # default False
        if True in binned_get:
            # XXX: this part depends on the concrete implementation of Binning, not a good idea
            try:
                binning_get = v.pop("binning")
            except KeyError:
                raise LoggedError(self.log, "binned=True, but missing binning")
            # different binning settings are not allowed
            if binning:
                kout = binning.pop("kout")
                try:
                    kout_ = binning_get.pop("kout")
                except KeyError:
                    raise LoggedError(self.log, "missing kout")
                if not (np.array_equal(kout, kout_) and binning == binning_get):
                    raise LoggedError(
                        self.log,
                        "does not support multiple different binning requirements",
                    )
                binning["kout"] = kout
            else:
                if "kout" not in binning_get:
                    raise LoggedError(self.log, "missing kout")
                binning = binning_get
        stored["binned"] = group_lists(binned, binned_get)
        stored["binning"] = binning

    def must_provide(self, **requirements):
        """
        ``nonlinear_Plk_grid={...}``: nonlinear power spectrum (Hubble unit) on grid, returns ``(ls, kgrid, Plk)``,
        ``ls`` are sorted in ascending order, but not necessarily match the one from requirements.
        Takes ``"ls": [list of evaluated multipoles]``; ``"chained": [False | True]`` compute chained power spectrum or not,
        by default False, if True, ``ls`` represents multipoles of chained power spectrum, if ``[False, True]``,
        ``ls`` represent multipoles of power spectrum; ``"binned": [False | True]``
        compute binned power spectrum or not, by default False, if True, extra
        ``"binning": {"kout": ndarray, ...}`` is required and returned ``kgrid``
        are effective ks

        ``nonlinear_Plk_interpolator={...}``: similar to ``nonlinear_Plk_grid``,
        but returns a cubic spline interpolator ``PlkInterpolator`` on ``kgrid``, does not support ``binned``

        ``nonlinear_Plk_gaussian_grid={...}``: similar to ``nonlinear_Plk_grid``,
        but returns the gaussian part of the power spectrum ``(ls, kgrid, PG_table)``
        where ``PG_table`` is a `dict[str, NDArray]` like dictionary, keys are gaussian parameters,
        values are Plk on grid.

        ``snapshots``: no settings required

        Notes
        -----
        all products should be treated as read-only, otherwise the cache may not work
        """

        def check_unsupported(k, v) -> None:
            if v:
                raise LoggedError(self.log, "unsupported requirements in %s: %s", k, v)

        super().must_provide(**requirements)
        reqs: dict[str, dict[str, Any]] = requirements.get(self.name + "_results", {})
        # TODO: support dynamical plugin settings
        for k, v in reqs.items():
            # nonlinear_Plk_interpolator based on nonlinear_Plk_grid
            if k in ("nonlinear_Plk_grid", "nonlinear_Plk_interpolator"):
                stored = self._must_provide[k]
                chained: list[bool] = stored.get("chained", [])
                chained_get = bool_or_list(v.pop("chained", False))  # default False
                stored["chained"] = group_lists(chained, chained_get)
                if k == "nonlinear_Plk_grid":
                    self._config_ls(k, v, chained=chained_get)
                else:
                    self._config_ls(
                        k, v, chained=chained_get, base="nonlinear_Plk_grid",
                    )
                    # request nonlinear_Plk_interpolator, but no nonlinear_Plk_grid requirements
                    base = self._must_provide["nonlinear_Plk_grid"]
                    chained_base: list[bool] = base.get("chained", [False])
                    base["chained"] = group_lists(chained_base, chained_get)
                    # require binned
                    binned_base: list[bool] = base.get("binned", [False])
                    base["binned"] = group_lists(binned_base, [False])
                    base.setdefault("binning", {})

                # nonlinear_Plk_interpolator does not support binned
                if k == "nonlinear_Plk_grid":
                    self._config_binned(k, v)

                check_unsupported(k, v)
            elif k == "nonlinear_Plk_gaussian_grid":
                stored = self._must_provide[k]
                chained: list[bool] = stored.get("chained", [])
                chained_get = bool_or_list(v.pop("chained", False))  # default False
                stored["chained"] = group_lists(chained, chained_get)
                self._config_ls(k, v, chained=chained_get)
                # I think different binning settings are rarely used
                self._config_binned("nonlinear_Plk_grid", v)
                check_unsupported(k, v)
            else:
                raise LoggedError(
                    self.log,
                    "Unexpected requirement %s, this should not happen, "
                    "please contact the developers",
                    k,
                )
        self.mpi_debug("updated must_provide: %s", self._must_provide)

    @cached_property
    def _params_reader(self):
        cross = self.config.get("cross", [])
        stnames = ("ce0", "cemono", "cequad")
        names = ("b1", "b2", "b3", "b4", "cct", "cr1", "cr2")
        if cross:
            A, B = (self.eftlss.tracers[name]["prefix"] for name in cross)

            def reader(params: dict[str, float]):
                es = [params[self.prefix + name] for name in stnames]
                bsA = [params[A + name] for name in names]
                bsB = [params[B + name] for name in names]
                b1A, b1B = bsA[0], bsB[0]
                return bsA, bsB, es, b1A, b1B

        else:

            def reader(params: dict[str, float]):
                es = [params[self.prefix + name] for name in stnames]
                bsA = [params[self.prefix + name] for name in names]
                bsB = bsA.copy()
                b1A, b1B = bsA[0], bsB[0]
                return bsA, bsB, es, b1A, b1B

        return reader

    @cached_property
    def _build_PG_table(self):
        """depending on BirdPlus.reducePG
        """
        prefix = self.prefix
        names = ("b3", "cct", "cr1", "cr2")
        stnames = ("ce0", "cemono", "cequad")
        alias = {prefix + k: k for k in stnames}
        cross = self.config.get("cross", [])
        if cross:
            A, B = (self.eftlss.tracers[name]["prefix"] for name in cross)
            for name in names:
                alias[A + name] = name + "A"
                alias[B + name] = name + "B"

            def builder(PG: dict[str, NDArray]) -> dict[str, NDArray]:
                out = {k: PG[v] for k, v in alias.items()}
                return {**PG, **out}

        else:

            def builder(PG: dict[str, NDArray]) -> dict[str, NDArray]:
                out = {k: PG[v] for k, v in alias.items()}
                for name in names:
                    out[prefix + name] = 2 * PG[name + "A"]
                return {**PG, **out}

        return builder

    @property
    def need_power(self):
        return any(
            item in self._must_provide
            for item in ("nonlinear_Plk_grid", "nonlinear_Plk_gaussian_grid")
        )

    @property
    def need_marg(self):
        return "nonlinear_Plk_gaussian_grid" in self._must_provide

    @property
    def need_binning(self):
        return True in self._must_provide.get("nonlinear_Plk_grid", {}).get(
            "binned", [False]
        )

    def calculate(self, state, want_derived=True, **params_values_dict: float):
        start = time.perf_counter()
        boltzmann = self.boltzmann
        # the main computation pipeline
        if self.need_power:
            bsA, bsB, es, b1A, b1B = self._params_reader(params_values_dict)
            if boltzmann.updated() or self.bird is None:
                # TODO: make kmin, kmax configurable
                kh = np.logspace(-5, 0, 200)
                pkh = boltzmann.Pkh(kh)
                H, DA, f = boltzmann.H, boltzmann.DA, boltzmann.f
                bird = BirdPlus(kh, pkh, f, DA, H, self.zeff, which="all", co=self.co)
                self.nonlinear.PsCf(bird)
                bird.setPsCfl()
                plugins = cast(PluginsDict, self.plugins)
                if self.with_IRresum:
                    plugins["IRresum"].Ps(bird)
                if self.with_APeffect:
                    plugins["APeffect"].AP(bird)
                if self.with_window:
                    plugins["window"].Window(bird)
                if self.with_fiber:
                    plugins["fiber"].fibcolWindow(bird)
                if self.binning:
                    self.binning.kbinning(bird)
                    bird.attach_hook(self.binning)
                bird.setreducePslb(bsA=bsA, bsB=bsB, es=es)
                if self.need_marg:
                    bird.setreducePG(b1A, b1B)
                self.bird = bird
            else:
                self.bird.setreducePslb(bsA=bsA, bsB=bsB, es=es)
                if self.need_marg:
                    self.bird.setreducePG(b1A, b1B)

        products: dict[str, Any] = {}
        if self.need_power:
            products["snapshots"] = self.bird.snapshots  # type: ignore
        # collect results
        for product, config_dict in self._must_provide.items():
            ls_tot = [2 * i for i in range(self.co.Nl)]
            kgrid = self.co.k
            if product == "nonlinear_Plk_grid":
                # XXX: necessary to perform the l cut?
                ls = config_dict["ls"]
                idx = [ls_tot.index(l) for l in ls]
                assert self.bird is not None
                results: dict[PlkKey, tuple[list[int], NDArray, NDArray]] = {}
                for chained, binned in itertools.product(
                    config_dict["chained"], config_dict["binned"]
                ):
                    if binned:
                        assert self.binning
                        plk = self.binning.fullPs
                        kreturn = self.binning.keff.copy()
                    else:
                        plk = self.bird.fullPs
                        kreturn = kgrid.copy()
                    if chained:
                        plk = to_chained(ls_tot, plk)
                        tup = (ls[:-1], kreturn, plk[idx[:-1]].copy())
                    else:
                        tup = (ls.copy(), kreturn, plk[idx].copy())
                    key = PlkKey(chained=chained, binned=binned)
                    results[key] = tup
                products[product] = results
                if "nonlinear_Plk_interpolator" in self._must_provide.keys():
                    reqs = self._must_provide["nonlinear_Plk_interpolator"]
                    results_interp: dict[PlkKey, PlkInterpolator] = {}
                    for chained in reqs["chained"]:
                        key = PlkKey(chained=chained, binned=False)
                        tup = products["nonlinear_Plk_grid"][key]
                        results_interp[key] = PlkInterpolator(*tup)
                    products["nonlinear_Plk_interpolator"] = results_interp

            elif product == "nonlinear_Plk_gaussian_grid":
                assert self.bird is not None
                out: dict[PlkKey, tuple[list[int], NDArray, dict[str, NDArray]]] = {}
                for chained, binned in itertools.product(
                    config_dict["chained"],
                    self._must_provide["nonlinear_Plk_grid"]["binned"],
                ):
                    if binned:
                        assert self.binning
                        PG = self.binning.PG
                        kreturn = self.binning.keff.copy()
                    else:
                        PG = self.bird.PG
                        kreturn = kgrid.copy()
                    PG = PG.copy()
                    if chained:
                        for kk, vv in PG.items():
                            PG[kk] = to_chained(ls_tot, vv)
                        PG_table = self._build_PG_table(PG)
                        tup = (ls_tot[:-1], kreturn, PG_table)
                    else:
                        PG_table = self._build_PG_table(PG)
                        tup = (ls_tot.copy(), kreturn, PG_table)
                    key = PlkKey(chained=chained, binned=binned)
                    out[key] = tup
                products[product] = out

        state[self.name + "_results"] = products

        if want_derived:
            plugins = cast(PluginsDict, self.plugins)
            if self.with_APeffect:
                # FIXME: return alperp, alpara
                try:
                    qperp, qpara = plugins["APeffect"].get_AP_param(self.bird)
                except KeyError:
                    if self.not_reported("no AP plugin"):
                        self.mpi_warning(
                            "APeffect not initialized, possiblely due to no power requested",
                        )
                else:
                    state["derived"][self.prefix + "alperp"] = qperp
                    state["derived"][self.prefix + "alpara"] = qpara
            else:
                state["derived"][self.prefix + "alperp"] = 1
                state["derived"][self.prefix + "alpara"] = 1
            state["derived"][self.prefix + "fz"] = boltzmann.f
            state["derived"][self.prefix + "fsigma8_z"] = boltzmann.fsigma8_z
            # TODO: fsigma8_cb_z
        end = time.perf_counter()
        self.mpi_debug("calculate: time used: %s", end - start)

    def get_can_provide(self):
        return [self.name + "_results"]

    def get_can_provide_params(self) -> list[str]:
        return [
            self.prefix + item
            for item in ("fz", "fsigma8_z", "fsigma8_cb_z", "alperp", "alpara")
        ]

    def not_reported(self, key: str) -> bool:
        flag = self._not_reported[key]
        self._not_reported[key] = False
        return flag
