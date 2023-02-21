from __future__ import annotations

# global
import itertools
import time
import numpy as np
from collections import defaultdict
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    cast,
    Iterable,
    NamedTuple,
    TypedDict,
    TYPE_CHECKING,
)
from scipy.interpolate import interp1d
from scipy.special import legendre
from cobaya.log import LoggedError
from cobaya.theory import HelperTheory
from cobaya.theory import Theory
from .parambasis import EastCoastBasis

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from cobaya.theory import Provider
    from .pybird.pybird import BirdSnapshot

# local
from .binning import Binning
from .icc import IntegralConstraint
from .interface import find_boltzmann_interface
from .parambasis import find_param_basis
from .pybird import pybird
from .tools import bool_or_list
from .tools import group_lists
from .tools import int_or_list
from .tools import Initializer
from .tools import recursively_update_dict


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
        if default := self.tracers.pop("default", {}):
            self.mpi_info("'default' field applies to all tracers")
            for k, tracer_config in self.tracers.items():
                default_config = deepcopy(default)
                recursively_update_dict(default_config, tracer_config)
                self.tracers[k] = default_config
        elif default := self.tracers.pop("common", {}):
            self.mpi_info("'common' field applies to all tracers")
            self.mpi_warning(
                "'common' field is deprecated, please use 'default' instead"
            )
            for k, tracer_config in self.tracers.items():
                default_config = deepcopy(default)
                recursively_update_dict(default_config, tracer_config)
                self.tracers[k] = default_config
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
            self.mpi_warning(
                "Seems like tracers are not sharing the same provider, is this intended?"
            )

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

        all requirements support ``default`` field

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
        # apply default settings
        if default := redirected_reqs_tmp.pop("default", {}):
            for tracer, config_dict in redirected_reqs.items():
                ref = deepcopy(default)
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
        for i, name in enumerate(self.names):
            # Pk_interpolator requires at least 4 redshift
            zextra = []
            if i == 0 and len(self.names) < 4:
                zeff = self.tracers[name]["z"]
                zextra = [zeff + i * 0.1 for i in range(1, 5 - len(out))]
            out["eftpipe.eftlss." + name] = EFTLSSLeaf(
                self,
                name,
                dict(stop_at_error=self.stop_at_error),
                timing=self.timer,  # type: ignore
                zextra=zextra,
            )
        return out


class PluginsDict(TypedDict):
    IRresum: pybird.Resum
    APeffect: pybird.APeffect
    window: pybird.Window
    icc: IntegralConstraint
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
        if len(idx) == 1:
            return self.fn(k)[idx][0]
        return self.fn(k)[idx]


class EFTLSSLeaf(HelperTheory):
    """EFT theory for single tracer

    Parameters
    ----------
    eftlss : EFTLSS
        parent EFTLSS theory
    name : str
        tracer's name
    info : dict
    timing : bool, optional
    zextra : list[float], optional
        append extra redshifts to satisfy Pk_interpolator's requirement
    """

    def __init__(
        self,
        eftlss: EFTLSS,
        name: str,
        info,
        timing: bool | None = None,
        zextra: list[float] | None = None,
    ) -> None:
        self._zextra: list[float] = zextra or []
        self.name = name
        self.eftlss = eftlss
        # EFTLSSChild always initialized after EFTLSS, safe to use config
        self.config = self.eftlss.tracers[name]
        super().__init__(info, self.eftlss.get_name() + "." + name, timing=timing)

    @cached_property
    def is_cross(self) -> bool:
        return bool(self.config.get("cross", False))

    @cached_property
    def cross_tracers(self) -> tuple[str, str]:
        return tuple(self.config.get("cross", ()))

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

    def setup_prefix(self) -> None:
        if (prefix := self.config.get("prefix")) is None:
            prefix = self.name + "_"
        self.prefix: str = prefix
        # write back
        self.config["prefix"] = prefix
        # find related prefix
        self.related_prefix: list[str] = []
        if cross := self.config.get("cross"):
            for name in cross:
                config = self.eftlss.tracers[name]
                self.related_prefix.append(
                    config["prefix"] if config.get("prefix") else name + "_"
                )

    def initialize(self) -> None:
        super().initialize()
        self.setup_prefix()
        with_NNLO = self.config.get("with_NNLO", False)
        self.basis = find_param_basis(self.config.get("basis", "westcoast"))(
            self.prefix, self.related_prefix, with_NNLO
        )
        self.mpi_info("EFT parameter basis: %s", self.basis.get_name())
        self.mpi_info("with_NNLO: %s", with_NNLO)
        self._not_reported = defaultdict(lambda: True)
        try:
            self.zeff: float = self.config["z"]
        except KeyError:
            raise LoggedError(self.log, "must specify effective redshift z")
        # provider hould be determined at initialization (requirements from boltzmann interface)
        self.boltzmann = find_boltzmann_interface(
            self.config.get("provider", "classy"),
            self.config.get("provider_kwargs", {}),
        )
        self.boltzmann.initialize(
            zeff=self.zeff, use_cb=self.config.get("use_cb", False), zextra=self._zextra
        )
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
            APeffect_config = deepcopy(self.config.get("APeffect", {}))
            if APeffect_config.get("z_AP") is None:
                APeffect_config["z_AP"] = self.zeff
            self.plugins["_APeffect"] = Initializer(
                pybird.APeffect,
                APeffect_config,
                self.log,
            )
        self.with_window: bool = self.config.get("with_window", False)
        if self.with_window:
            self.plugins["_window"] = Initializer(
                pybird.Window, self.config.get("window", {}), self.log
            )
        self.with_fiber: bool = self.config.get("with_fiber", False)
        if self.with_fiber:
            self.plugins["_fiber"] = Initializer(
                pybird.FiberCollision, self.config.get("fiber", {}), self.log
            )
        self.with_icc: bool = self.config.get("with_icc", False)
        if self.with_icc:
            self.plugins["_icc"] = Initializer(
                IntegralConstraint, self.config.get("icc", {}), self.log
            )
        self.binning: Binning | None = None
        self._must_provide: defaultdict[str, dict[str, Any]] = defaultdict(dict)

    def extract_km_kr_nd(self):
        cross_tracers = self.cross_tracers
        if cross_tracers:
            try:
                tracerA, tracerB = (self.eftlss.tracers[_] for _ in cross_tracers)
                kmA, krA, ndA = tracerA["km"], tracerA.get("kr"), tracerA["nd"]
                kmB, krB, ndB = tracerB["km"], tracerB.get("kr"), tracerB["nd"]
                if krA is None:
                    self.mpi_warning(
                        "%s: kr not specified, assuming kr=km, will raise exception in the future",
                        cross_tracers[0],
                    )
                    krA = kmA
                if krB is None:
                    self.mpi_warning(
                        "%s: kr not specified, assuming kr=km, will raise exception in the future",
                        cross_tracers[1],
                    )
                    krB = kmB
            except KeyError:
                raise LoggedError(
                    self.log, "missing km, kr or nd for tracer %s", cross_tracers
                )
        else:
            try:
                kmA, krA, ndA = (
                    self.config["km"],
                    self.config.get("kr"),
                    self.config["nd"],
                )
                if krA is None:
                    self.mpi_warning(
                        "kr not specified, assuming kr=km, will raise exception in the future"
                    )
                    krA = kmA
                kmB, krB, ndB = kmA, krA, ndA
            except KeyError:
                raise LoggedError(self.log, "must specify km, kr and nd")
        if isinstance(self.basis, EastCoastBasis) and kmA != krA:
            raise LoggedError(self.log, "eastcoast basis is usually used with km=kr")
        return kmA, krA, ndA, kmB, krB, ndB

    def initialize_with_provider(self, provider: Provider):
        super().initialize_with_provider(provider)

        self.boltzmann.initialize_with_provider(provider)
        kmA, krA, ndA, kmB, krB, ndB = self.extract_km_kr_nd()

        # do not initialize if no power spectrum requested
        if not self.need_power:
            return
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        counterform = self.config.get("counterform") or self.basis.counterform()
        if counterform != self.basis.counterform():
            self.mpi_warning(
                "specified counterform %s is different from the one in basis %s",
                counterform,
                self.basis.counterform(),
            )
        if self.cross_tracers:
            _A, _B = self.cross_tracers
            self.mpi_info("%s: km=%.3f kr=%.3f nd=%.3e", _A, kmA, krA, ndA)
            self.mpi_info("%s: km=%.3f kr=%.3f nd=%.3e", _B, kmB, krB, ndB)
        else:
            self.mpi_info("km=%.3f kr=%.3f nd=%.3e", kmA, krA, ndA)
        self.mpi_info("effective redshift: %.3f", self.zeff)
        if self.cross_tracers:
            self.mpi_info(
                "computing cross power spectrum between %s and %s", *self.cross_tracers
            )
        ls_group = []
        for name in (
            "nonlinear_Plk_grid",
            "nonlinear_Plk_interpolator",
            "nonlinear_Plk_gaussian_grid",
        ):
            if name not in self._must_provide:
                continue
            ls_group.append(self._must_provide[name]["ls"])
        # pybird actually does not support computing, e.g. P4 only
        lmax = max(group_lists(*ls_group))
        ls = list(range(0, lmax + 2, 2))
        Nl = len(ls)
        self.mpi_info("compute power spectrum multipoles (internal): %s", ls)

        optiresum: bool = self.config.get("IRresum", {}).pop("optiresum", False)
        if optiresum:
            self.mpi_warning(
                "no test on ``optiresum = true``, use it only if you know what you are doing"
            )
        self.co = pybird.Common(
            Nl=Nl,
            optiresum=optiresum,
            kmA=kmA,
            krA=krA,
            ndA=ndA,
            kmB=kmB,
            krB=krB,
            ndB=ndB,
            counterform=counterform,
            with_NNLO=self.config.get("with_NNLO", False),
        )
        self.bird: pybird.Bird | None = None
        self.nonlinear = pybird.NonLinear(
            load=True,
            save=True,
            co=self.co,
            path=str(self.eftlss.cache_dir_path),
            name=self.get_name() + ".nonlinear",
        )

        if self.config.get("use_cb", False):
            self.mpi_info("using P_cb as input linear power spectrum")
        else:
            self.mpi_info("using P_m as input linear power spectrum")

        msg_pool = []
        if self.with_IRresum:
            self.plugins["IRresum"] = self.plugins["_IRresum"].initialize(
                co=self.co, name=self.get_name() + ".IRresum"
            )
            msg_pool.append(
                ("IRresum enabled: %s", "optimized" if optiresum else "full")
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
        if self.with_icc:
            if self.is_cross:
                raise LoggedError(
                    self.log,
                    "integral constraint correction (icc) not yet supported for cross power spectrum",
                )
            self.plugins["icc"] = self.plugins["_icc"].initialize(
                co=self.co, name=self.get_name() + ".icc"
            )
            msg_pool.append(("Integral Constraint Correction (icc) enabled",))
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
        requires = self.boltzmann.get_requirements()
        eft_params = {param: None for param in self.basis.non_gaussian_params()}
        requires.update(eft_params)
        return requires

    def get_can_support_params(self):
        return self.basis.gaussian_params()

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
                        k,
                        v,
                        chained=chained_get,
                        base="nonlinear_Plk_grid",
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
                # TODO: ``requires`` option to select which gaussian parameters are needed
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

    def calculate(self, state, want_derived=True, **params_values_dict: float):
        start = time.perf_counter()
        boltzmann = self.boltzmann
        # the main computation pipeline
        if self.need_power:
            if boltzmann.updated() or self.bird is None:
                # TODO: make kmin, kmax configurable
                kh = np.logspace(-5, 0, 200)
                pkh = boltzmann.Pkh(kh)
                H, DA, f = boltzmann.H(), boltzmann.DA(), boltzmann.f()
                rdrag, h = boltzmann.rdrag(), boltzmann.h()
                bird = pybird.Bird(
                    kh, pkh, f, DA, H, self.zeff, co=self.co, rdrag=rdrag, h=h
                )
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
                if self.with_icc:
                    plugins["icc"].icc(bird)
                if self.binning:
                    self.binning.kbinning(bird)
                    bird.attach_hook(self.binning)
                self.bird = bird
            # use basis to compute power spectrum
            self.basis.reduce_Pk(self.bird, params_values_dict)

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
                        PG_table = self.basis.create_binned_PG_table(
                            self.binning,
                            self.bird,
                            params_values_dict,
                        )
                        kreturn = self.binning.keff.copy()
                    else:
                        PG_table = self.basis.create_PG_table(
                            self.bird,
                            params_values_dict,
                        )
                        kreturn = kgrid.copy()
                    if chained:
                        for kk, vv in PG_table.items():
                            PG_table[kk] = to_chained(ls_tot, vv)
                        tup = (ls_tot[:-1], kreturn, PG_table)
                    else:
                        tup = (ls_tot.copy(), kreturn, PG_table)
                    key = PlkKey(chained=chained, binned=binned)
                    out[key] = tup
                products[product] = out

        state[self.name + "_results"] = products

        if want_derived:
            plugins = cast(PluginsDict, self.plugins)
            if self.with_APeffect:
                try:
                    alperp, alpara = plugins["APeffect"].get_alperp_alpara(self.bird)
                except KeyError:
                    if self.not_reported("no AP plugin"):
                        self.mpi_warning(
                            "APeffect not initialized, possiblely due to no power requested",
                        )
                else:
                    state["derived"][self.prefix + "alperp"] = alperp
                    state["derived"][self.prefix + "alpara"] = alpara
            else:
                state["derived"][self.prefix + "alperp"] = -1
                state["derived"][self.prefix + "alpara"] = -1
            state["derived"][self.prefix + "fz"] = boltzmann.f()
            state["derived"][self.prefix + "fsigma8_z"] = boltzmann.fsigma8_z()
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
