from __future__ import annotations
import importlib
import functools
import itertools
import numpy as np
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from dataclasses import InitVar
from pathlib import Path
from time import perf_counter
from typing import (
    Any,
    cast,
    Callable,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    TypedDict,
    TYPE_CHECKING,
)
from cobaya.log import LoggedError
from cobaya.theory import HelperTheory
from cobaya.theory import Theory
from cobaya.typing import empty_dict, TheoryDictIn
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from cobaya.theory import Provider
    from numpy.typing import ArrayLike
    from .boltzmann import BoltzmannInterface
    from .pybird.pybird import BirdLike
    from .pybird.pybird import BirdSnapshot
    from .parambasis import BirdComponent, EFTBasis
    from .typing import ndarrayf

from .binning import Binning
from .chained import Chained
from .icc import IntegralConstraint
from .boltzmann import find_boltzmann_interface
from .parambasis import find_param_basis
from .pybird import pybird
from .tools import bool_or_list
from .tools import group_lists
from .tools import int_or_list
from .tools import Initializer
from .tools import recursively_update_dict
from .transformer import BirdCopier
from .window import Window


def leaf_product_name(tracer: str):
    return f"eftleaf_{tracer}_results"


def leaf_kernel_product_name(tracer: str):
    return f"eftleaf_kernel_{tracer}_results"


def find_window_constructor(name: str) -> Callable:
    if name == "auto" or name == "default":
        return Window
    module_name, class_name = name.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module_name, tmp = module_name.rsplit(".", 1)
        class_name = f"{tmp}.{class_name}"
        module = importlib.import_module(module_name)
    return functools.reduce(getattr, class_name.split("."), module)


@dataclass(eq=False)
class PlkInterpolator:
    ls: list[int]
    kgrid: InitVar[ndarrayf]
    Plk: InitVar[ndarrayf]

    fn: Callable[[ArrayLike], ndarrayf] = field(init=False, repr=False)

    def __post_init__(self, kgrid: ndarrayf, Plk: ndarrayf):
        self.ls = self.ls.copy()
        # XXX: not sure if inserting zero is a good idea
        kgrid = np.hstack(([0], kgrid))
        Plk = np.insert(Plk, 0, 0, axis=-1)
        tmp = interp1d(
            kgrid,
            kgrid * Plk,
            axis=-1,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",  # type: ignore
        )
        self.fn = lambda k: tmp(k) / k

    def __call__(self, l: int | Iterable[int], k: ArrayLike) -> ndarrayf:
        l = int_or_list(l)
        try:
            idx = [self.ls.index(ll) for ll in l]
        except ValueError as ex:
            raise ValueError(f"l={l} not in {self.ls}") from ex
        if len(idx) == 1:
            return self.fn(k)[idx][0]
        return self.fn(k)[idx]


class PluginsDict(TypedDict):
    IRresum: pybird.Resum
    APeffect: pybird.APeffect
    window: Window
    fiber: pybird.FiberCollision


class EFTLSS(Theory):
    """Effective Field Theory of Large-scale Structures"""

    cache_dir_path: Path
    tracers: dict[str, dict[str, Any]]
    # override default
    file_base_name = "eftlss"
    # other attrs
    tracer_names: Sequence[str]

    def initialize(self) -> None:
        # step 1: check and apply 'default' field to all tracers
        self.cache_dir_path = Path(self.cache_dir_path)
        if not self.cache_dir_path.is_dir():
            self.cache_dir_path.mkdir(parents=True)
        # XXX: self.tracers seems not completely copied by cobaya, manually copy it here
        self.tracers = deepcopy(self.tracers)
        if default := self.tracers.pop("default", {}):
            self.mpi_info("'default' field applies to all tracers")
            for tracer_name, config in self.tracers.items():
                default_config = deepcopy(default)
                recursively_update_dict(default_config, config)
                self.tracers[tracer_name] = default_config
        self.tracer_names = list(self.tracers.keys())
        if not self.tracer_names:
            raise LoggedError(self.log, "No tracer specified")
        # step 2: check 'cross' field is valid
        for tracer_name, config in self.tracers.items():
            cross = config.get("cross", False)
            if isinstance(cross, bool):
                continue
            if not isinstance(cross, list) or len(cross) != 2:
                msg = "tracer %s: expect a list of 2 elements, but given cross=%r"
                raise LoggedError(self.log, msg, tracer_name, cross)
            if diff := set(cross).difference(self.tracer_names):
                msg = "tracer %s: cross=%r contains unknown tracer names: %r"
                raise LoggedError(self.log, msg, tracer_name, cross, diff)
        # step 3: check if tracers are using the same 'boltzmann' provider
        providers = set(
            config.get("provider", "classy") for config in self.tracers.values()
        )
        if len(providers) != 1:
            self.mpi_warning(
                "Tracers are not sharing the same provider, is this intended?"
            )

    def get_requirements(self):
        """
        dummy requirements: make it possible to use eftlss with likelihood one,
        otherwise eftlss does not depend on anything
        """
        return {leaf_product_name(tracer): {} for tracer in self.tracer_names}

    def must_provide(
        self, **requirements: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """redirect requirements to EFTLeaf and EFTLeafKernel

        For example, ``nonlinear_Plk_grid={{'LRG': {...}, 'ELG': {...}}}`` will
        be packed like ``{'eftleaf_LRG_results': {'nonlinear_Plk_grid': {...}}, 'eftleaf_ELG_results': {'nonlinear_Plk_grid': {...}}}``

        all requirements support ``default`` field

        detailed requirement form can be found in ``EFTLeaf.must_provide``
        """
        super().must_provide(**requirements)
        redirected: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        for product, config_per_tracer in requirements.items():
            config_per_tracer: dict[str, Any]
            for tracer, config in config_per_tracer.items():
                if (tracer != "default") and (tracer not in self.tracer_names):
                    raise LoggedError(self.log, "Unknown tracer name: %s", tracer)
                redirected[tracer][product] = config
        reqs = {}
        for tracer, product_config in redirected.items():
            reqs[leaf_product_name(tracer)] = deepcopy(product_config)
            reqs[leaf_kernel_product_name(tracer)] = deepcopy(product_config)
        # apply default settings
        if default := reqs.pop("default", {}):
            for tracer, config in reqs.items():
                default_config = deepcopy(default)
                recursively_update_dict(default_config, config)
                reqs[tracer] = default_config
        self.mpi_debug("Redirected requirements: %r", reqs)
        return reqs

    def get_helper_theories(self) -> dict[str, Theory]:
        info = {"stop_at_error": self.stop_at_error}
        helpers = {}
        for i, tracer in enumerate(self.tracer_names):
            # Pk_interpolator requires at least 4 redshift
            zextra = []
            if i == 0 and len(self.tracer_names) < 4:
                zeff = self.tracers[tracer]["z"]
                zextra = [zeff + i * 0.1 for i in range(1, 5 - len(helpers))]
            # leaf
            leaf_name = self.leaf_name(tracer)
            helpers[leaf_name] = EFTLeaf(
                info=info,
                name=leaf_name,
                timing=self.timer,
                tracer=tracer,
                eftlss=self,
            )
            # kernel
            leaf_kernel_name = self.leaf_kernel_name(tracer)
            helpers[leaf_kernel_name] = EFTLeafKernel(
                info=info,
                name=leaf_kernel_name,
                timing=self.timer,
                tracer=tracer,
                eftlss=self,
                zextra=zextra.copy(),
            )
        return helpers

    def leaf_name(self, tracer: str):
        return self.get_name() + "." + tracer

    def leaf_kernel_name(self, tracer: str):
        return self.leaf_name(tracer) + ".kernel"

    def retrieve_product_from_leaf(self, tracer: str, product: Any):
        if tracer not in self.tracer_names:
            raise ValueError(f"Tracer {tracer} not in {self.tracer_names}!")
        try:
            return self.provider.get_result(leaf_product_name(tracer))[product]
        except KeyError:
            raise LoggedError(
                self.log,
                "%s not computed, please check if you have specified it in requirements",
                product,
            )

    def get_nonlinear_Plk_grid(
        self, tracer: str, chained: bool = False, binned: bool = False
    ) -> tuple[list[int], ndarrayf, ndarrayf]:
        key = ("nonlinear_Plk_grid", chained, binned)
        return self.retrieve_product_from_leaf(tracer, key)

    def get_nonlinear_Plk_gaussian_grid(
        self, tracer: str, chained: bool = False, binned: bool = False
    ) -> tuple[list[int], ndarrayf, dict[str, ndarrayf]]:
        key = ("nonlinear_Plk_gaussian_grid", chained, binned)
        return self.retrieve_product_from_leaf(tracer, key)

    def get_nonlinear_Plk_interpolator(
        self, tracer: str, chained: bool = False
    ) -> PlkInterpolator:
        key = ("nonlinear_Plk_interpolator", chained)
        return self.retrieve_product_from_leaf(tracer, key)

    def get_snapshots(self, tracer: str) -> dict[str, BirdSnapshot]:
        return self.retrieve_product_from_leaf(tracer, "snapshots")

    def get_eft_params_values_dict(self, tracer: str) -> dict[str, float]:
        return self.retrieve_product_from_leaf(tracer, "eft_params_values_dict")

    def get_bird_component(
        self, tracer: str
    ) -> tuple[list[int], ndarrayf, BirdComponent]:
        return self.retrieve_product_from_leaf(tracer, "bird_component")


class LeafKernelShared(Protocol):
    tracer: str
    tracer_prefix: str
    tracer_config: Mapping[str, Any]
    eftlss: EFTLSS

    def set_tracer_prefix(self):
        if (prefix := self.tracer_config.get("prefix")) is None:
            prefix = self.tracer + "_"
        self.tracer_prefix = prefix

    def cross_type(self) -> bool | list[str]:
        return self.tracer_config.get("cross", False)

    def build_basis(self) -> EFTBasis:
        if (prefix := self.tracer_config.get("prefix")) is None:
            prefix = self.tracer + "_"
        related_prefix: list[str] = []
        if isinstance(cross := self.cross_type(), Iterable):
            for name in cross:
                config = self.eftlss.tracers[name]
                related_prefix.append(
                    config["prefix"] if config.get("prefix") else name + "_"
                )
        return find_param_basis(self.tracer_config.get("basis", "westcoast"))(
            prefix=prefix, cross_prefix=related_prefix
        )


class EFTLeafKernel(HelperTheory, LeafKernelShared):
    """EFT theory for single tracer (EFT parameterization independent part)"""

    tracer: str
    tracer_prefix: str
    tracer_config: Mapping[str, Any]
    eftlss: EFTLSS
    zextra: list[float]
    basis: EFTBasis
    zeff: float
    boltzmann: BoltzmannInterface
    plugins: dict[str, Any]

    def __init__(
        self,
        info: TheoryDictIn = empty_dict,
        name: str | None = None,
        timing: Any | None = None,
        packages_path: str | None = None,
        initialize: bool = True,
        standalone: bool = True,
        tracer: str = "",
        eftlss: EFTLSS | None = None,
        zextra: list[float] = [],
    ):
        self.tracer = tracer
        self.eftlss = cast(EFTLSS, eftlss)
        self.zextra = zextra or []
        # EFTLeafKernel always initialized after EFTLSS, safe to use config
        self.tracer_config = self.eftlss.tracers[tracer]
        super().__init__(
            info=info,
            name=name,
            timing=timing,
            packages_path=packages_path,
            initialize=initialize,
            standalone=standalone,
        )

    def initialize(self):
        super().initialize()
        self.set_tracer_prefix()
        self.basis = self.build_basis()
        self.zeff = self.tracer_config["z"]
        self.boltzmann = find_boltzmann_interface(
            self.tracer_config.get("provider", "classy"),
            self.tracer_config.get("provider_kwargs", {}),
        )
        self.boltzmann.initialize(
            zeff=self.zeff,
            use_cb=self.tracer_config.get("use_cb", False),
            zextra=self.zextra,
        )
        # delayed initialization in initialize_with_provider,
        # so that all dependencies can be quickly checked
        # check config
        self.plugins: dict[str, Any] = {}
        self.with_RSD: bool = self.tracer_config.get("with_RSD", True)
        self.with_IRresum: bool = self.tracer_config.get("with_IRresum", True)
        if self.with_IRresum:
            self.plugins["_IRresum"] = Initializer(
                pybird.Resum, self.tracer_config.get("IRresum", {}), self.log
            )
        self.with_APeffect: bool = self.tracer_config.get("with_APeffect", False)
        if self.with_APeffect:
            APeffect_config = deepcopy(self.tracer_config.get("APeffect", {}))
            if APeffect_config.get("z_AP") is None:
                APeffect_config["z_AP"] = self.zeff
            self.plugins["_APeffect"] = Initializer(
                pybird.APeffect,
                APeffect_config,
                self.log,
            )
        self.with_window: bool | str = self.tracer_config.get("with_window", False)
        if self.with_window:
            window_name: str = "auto" if self.with_window is True else self.with_window
            self.plugins["_window"] = Initializer(
                find_window_constructor(window_name),
                self.tracer_config.get("window", {}),
                self.log,
            )
        self.with_fiber: bool = self.tracer_config.get("with_fiber", False)
        if self.with_fiber:
            self.plugins["_fiber"] = Initializer(
                pybird.FiberCollision, self.tracer_config.get("fiber", {}), self.log
            )
        # TODO: move icc config to window
        self.with_icc: bool = self.tracer_config.get("with_icc", False)
        if self.with_icc:
            self.plugins["_icc"] = Initializer(
                IntegralConstraint, self.tracer_config.get("icc", {}), self.log
            )
        # allowed keys: Nl, binned, binning, chained, bird_component
        self._must_provide: dict[str, Any] = {
            "Nl": self.tracer_config.get("Nl", 0) or 0,
            "No": 0,
            "binned": [],
            "chained": [],
            "bird_component": False,
        }
        self._warned = set()

    def initialize_with_provider(self, provider: Provider):
        super().initialize_with_provider(provider)
        self.boltzmann.initialize_with_provider(provider)
        # do not initialize if no power spectrum is required
        if not self.required_power_spectrum():
            return

        self.mpi_info("EFT parameter basis: %s", self.basis.get_name())
        self.mpi_info("with_NNLO: %s", self.tracer_config.get("with_NNLO", False))
        counterform = self.tracer_config.get("counterform") or self.basis.counterform()
        if counterform != self.basis.counterform():
            self.mpi_warning(
                "specified counterform %s is different from the one in basis %s",
                counterform,
                self.basis.counterform(),
            )
        self.report_cross_type()
        self.mpi_info(
            "compute power spectrum multipoles (internally): %s",
            [2 * i for i in range(self.Nl())],
        )

        optiresum: bool = self.tracer_config.get("optiresum", False)
        kmA, krA, ndA, kmB, krB, ndB = self.extract_km_kr_nd()
        self.co = pybird.Common(
            Nl=self.Nl(),
            No=self.No(),
            kmax=self.tracer_config.get("kmax", 0.3),
            optiresum=optiresum,
            kmA=kmA,
            krA=krA,
            ndA=ndA,
            kmB=kmB,
            krB=krB,
            ndB=ndB,
            counterform=counterform,
            with_NNLO=self.tracer_config.get("with_NNLO", False),
            IRcutoff=self.tracer_config.get("IRcutoff", False),
            kIR=self.tracer_config.get("kIR", None),
        )
        self.nonlinear = pybird.NonLinear(
            load=True,
            save=True,
            co=self.co,
            path=str(self.eftlss.cache_dir_path),
            name=self.get_name() + ".nonlinear",
        )

        # initialize plugins
        msg_pool = []
        if self.co.IRcutoff:
            msg_pool.append(
                ("IRcutoff enabled: mode=%s, kIR = %.6f", self.co.IRcutoff, self.co.kIR)
            )
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
        if self.with_icc:
            if self.cross_type():
                raise LoggedError(
                    self.log,
                    "integral constraint correction (icc) not yet supported for cross power spectrum",
                )
            self.plugins["icc"] = self.plugins["_icc"].initialize(
                co=self.co, name=self.get_name() + ".icc"
            )
        if self.with_window:
            self.plugins["window"] = self.plugins["_window"].initialize(
                co=self.co,
                icc=self.plugins.get("icc"),
                name=self.get_name() + ".window",
            )
            msg_pool.append(("window enabled",))
            if self.plugins.get("icc"):
                msg_pool.append(("Integral Constraint Correction (icc) enabled",))
        if self.with_fiber:
            self.plugins["fiber"] = self.plugins["_fiber"].initialize(
                co=self.co, name=self.get_name() + ".fiber"
            )
            msg_pool.append(("fiber enabled",))
        for msg in msg_pool:
            self.mpi_info(*msg)

        if self.required_binning():
            self.binning = Binning(
                **self._must_provide["binning"],
                co=self.co,
                name=self.get_name() + ".binning",
            )
        self.chained_tranformer = Chained()

    def get_requirements(self):
        return self.boltzmann.get_requirements()

    def must_provide(self, **requirements):
        super().must_provide(**requirements)
        reqs: dict[str, dict[str, Any]] = requirements.get(self.product_name(), {})
        # TODO: support dynamical plugin settings
        for product, config in reqs.items():
            if product == "bird_component":
                self._must_provide["bird_component"] = True
            elif product in (
                "nonlinear_Plk_grid",
                "nonlinear_Plk_interpolator",
                "nonlinear_Plk_gaussian_grid",
            ):
                # step 1: determine maximum Nl
                ls = int_or_list(config["ls"])  # mandatory key in config
                if bool_or_list(config.get("chained", False)) == [True]:
                    ls = list(set(ls + [l + 2 for l in ls]))
                if any(l % 2 == 1 for l in ls):
                    raise LoggedError(self.log, "Invalid multipoles: %s", ls)
                if max(ls) > 4:
                    raise LoggedError(self.log, "Unsupported multipoles: %s", ls)
                Nl = max(ls) // 2 + 1
                self._must_provide["Nl"] = max(Nl, self._must_provide["Nl"])
                self._must_provide["No"] = max(Nl, self._must_provide["No"])
                # step 2: configure chained
                self._must_provide["chained"] = group_lists(
                    bool_or_list(config.get("chained", False)),
                    self._must_provide["chained"],
                )
                # step 3: safely configure binning
                binned = bool_or_list(config.get("binned", False))
                if True in binned:
                    try:
                        binning = config["binning"]
                    except KeyError:
                        raise LoggedError(self.log, "binned=True but missing binning")
                    # XXX: this part depends on the concrete implementation of Binning, not a good idea
                    try:
                        kout = binning["kout"]
                    except KeyError:
                        raise LoggedError(self.log, "missing kout in binning")
                    if stored_binning := self._must_provide.get("binning"):
                        # different binning settings are not allowed
                        stored_kout = stored_binning.pop("kout")
                        binning_wo_kout = {
                            k: v for k, v in binning.items() if k != "kout"
                        }
                        compatible = (
                            np.array_equal(kout, stored_kout)
                            and binning_wo_kout == stored_binning
                        )
                        if not compatible:
                            raise LoggedError(
                                self.log,
                                "does not support multiple different binning requirements",
                            )
                    self._must_provide["binning"] = deepcopy(binning)
                self._must_provide["binned"] = group_lists(
                    binned, self._must_provide["binned"]
                )
                assert self._must_provide["binned"]  # not possible to be empty
        self.mpi_debug("updated must_provide: %s", self._must_provide)

    def calculate_power_spectrum(self, state, want_derived=True, **params_values_dict):
        # step 1: bird
        boltzmann = self.boltzmann
        # TODO: make kmin, kmax configurable
        kh = np.logspace(-5, 0, 200)
        pkh = boltzmann.Pkh(kh)
        H, DA, f = boltzmann.H(), boltzmann.DA(), boltzmann.f()
        rdrag, h = boltzmann.rdrag(), boltzmann.h()
        if not self.with_RSD:
            f = 0.0
        bird = pybird.Bird(kh, pkh, f, DA, H, self.zeff, co=self.co, rdrag=rdrag, h=h)
        self.nonlinear.PsCf(bird)
        bird.setPsCfl()
        self.bird = bird
        plugins = cast(PluginsDict, self.plugins)
        if self.with_IRresum:
            plugins["IRresum"].Ps(bird)
        bird_component_product = None
        if self._must_provide["bird_component"]:
            bird_component_product = (
                [2 * i for i in range(self.Nl())],
                bird.co.k.copy(),
                BirdCopier().transform(bird),
            )
        if self.with_APeffect:
            plugins["APeffect"].AP(bird)
        if self.with_window:
            plugins["window"].Window(bird)
        if self.with_fiber:
            plugins["fiber"].fibcolWindow(bird)
        # step 2: collect birdlike products
        birdlike_products: dict[
            tuple[str, str], tuple[list[int], ndarrayf, BirdLike]
        ] = {}
        for chained, binned in itertools.product(
            self._must_provide["chained"], self._must_provide["binned"]
        ):
            birdlike = bird
            if binned:
                kreturn = self.binning.keff.copy()
                birdlike = self.binning.transform(birdlike)
            else:
                kreturn = bird.co.k.copy()
            if chained:
                ls = [2 * l for l in range(self.No() - 1)]
                birdlike = self.chained_tranformer.transform(birdlike)
            else:
                ls = [2 * l for l in range(self.No())]
            birdlike_products[(chained, binned)] = (ls, kreturn, birdlike)
        state[self.product_name()] = {
            "bird": bird,
            "birdlike_products": birdlike_products,
            "bird_component_product": bird_component_product,
        }

    def calculate(self, state, want_derived=True, **params_values_dict):
        tstart = perf_counter()
        # empty if not required_power_spectrum
        state[self.product_name()] = {}
        if self.required_power_spectrum():
            self.calculate_power_spectrum(state, want_derived, **params_values_dict)

        if want_derived:
            boltzmann = self.boltzmann
            plugins = cast(PluginsDict, self.plugins)
            if self.with_APeffect:
                try:
                    alperp, alpara = plugins["APeffect"].get_alperp_alpara(self.bird)
                except (KeyError, AttributeError):
                    if self.not_warned("no AP plugin"):
                        self.mpi_warning(
                            "APeffect not initialized, possiblely due to no power spectrum requested"
                        )
                else:
                    state["derived"][self.tracer_prefix + "alperp"] = alperp
                    state["derived"][self.tracer_prefix + "alpara"] = alpara
            else:
                state["derived"][self.tracer_prefix + "alperp"] = -1
                state["derived"][self.tracer_prefix + "alpara"] = -1
            state["derived"][self.tracer_prefix + "fz"] = boltzmann.f()
            state["derived"][self.tracer_prefix + "fsigma8_z"] = boltzmann.fsigma8_z()
            # TODO: fsigma8_cb_z
        tend = perf_counter()
        self.mpi_debug("calculate %s: %fs", self.product_name(), tend - tstart)

    def get_can_provide(self):
        return [self.product_name()]

    def get_can_provide_params(self):
        return [
            self.tracer_prefix + item
            for item in ("fz", "fsigma8_z", "fsigma8_cb_z", "alperp", "alpara")
        ]

    def report_cross_type(self):
        cross = self.cross_type()
        kmA, krA, ndA, kmB, krB, ndB = self.extract_km_kr_nd()
        if isinstance(cross, bool):
            self.mpi_info("km=%s, kr=%s, nd=%s", kmA, krA, ndA)
            if cross:
                self.mpi_info("using cross itself's parameterization")
        else:
            tracerA, tracerB = cross
            self.mpi_info("cross correlation between %s and %s", tracerA, tracerB)
            self.mpi_info("%s: km=%s, kr=%s, nd=%s", tracerA, kmA, krA, ndA)
            self.mpi_info("%s: km=%s, kr=%s, nd=%s", tracerB, kmB, krB, ndB)

    def extract_km_kr_nd(self):
        cross_type = self.cross_type()
        if isinstance(cross_type, bool):
            try:
                kmA, krA, ndA = (
                    self.tracer_config["km"],
                    self.tracer_config.get("kr"),
                    self.tracer_config["nd"],
                )
                if krA is None:
                    self.mpi_warning(
                        "kr not specified, assuming kr=km, will raise exception in the future"
                    )
                    krA = kmA
                kmB, krB, ndB = kmA, krA, ndA
            except KeyError:
                raise LoggedError(self.log, "must specify km, kr and nd")
        else:
            try:
                tracerA, tracerB = (self.eftlss.tracers[_] for _ in cross_type)
                kmA, krA, ndA = tracerA["km"], tracerA.get("kr"), tracerA["nd"]
                kmB, krB, ndB = tracerB["km"], tracerB.get("kr"), tracerB["nd"]
                if krA is None:
                    self.mpi_warning(
                        "%s: kr not specified, assuming kr=km, will raise exception in the future",
                        cross_type[0],
                    )
                    krA = kmA
                if krB is None:
                    self.mpi_warning(
                        "%s: kr not specified, assuming kr=km, will raise exception in the future",
                        cross_type[1],
                    )
                    krB = kmB
            except KeyError:
                raise LoggedError(
                    self.log, "missing km, kr or nd for tracer %s", cross_type
                )
        return kmA, krA, ndA, kmB, krB, ndB

    def product_name(self) -> str:
        return leaf_kernel_product_name(self.tracer)

    def Nl(self) -> int:
        return self._must_provide["Nl"]

    def No(self) -> int:
        return self._must_provide["No"]

    def required_power_spectrum(self) -> bool:
        return self.No() != 0

    def required_binning(self) -> bool:
        return True in self._must_provide["binned"]

    def not_warned(self, key: str) -> bool:
        if key in self._warned:
            return False
        self._warned.add(key)
        return True


class EFTLeaf(HelperTheory, LeafKernelShared):
    """
    EFT theory for single tracer
    """

    tracer: str
    tracer_prefix: str
    tracer_config: Mapping[str, Any]
    eftlss: EFTLSS
    basis: EFTBasis

    def __init__(
        self,
        info: TheoryDictIn = empty_dict,
        name: str | None = None,
        timing: Any | None = None,
        packages_path: str | None = None,
        initialize: bool = True,
        standalone: bool = True,
        tracer: str = "",
        eftlss: EFTLSS | None = None,
    ):
        self.tracer = tracer
        self.eftlss = cast(EFTLSS, eftlss)
        # EFTLeaf always initialized after EFTLSS, safe to use config
        self.tracer_config = self.eftlss.tracers[tracer]
        super().__init__(
            info=info,
            name=name,
            timing=timing,
            packages_path=packages_path,
            initialize=initialize,
            standalone=standalone,
        )

    def initialize(self):
        super().initialize()
        self.set_tracer_prefix()
        self.basis = self.build_basis()
        self._must_provide: dict[str, set[tuple[bool, bool]]] = {
            "nonlinear_Plk_grid": set(),
            "nonlinear_Plk_interpolator": set(),
            "nonlinear_Plk_gaussian_grid": set(),
        }

    def get_requirements(self):
        requires: dict[str, Any] = {k: None for k in self.basis.non_gaussian_params()}
        requires[self.kernel_product_name()] = {}
        return requires

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

        ``eft_params_values_dict``: no settings required

        Notes
        -----
        all products should be treated as read-only, otherwise the cache may not work
        """
        super().must_provide(**requirements)
        reqs: dict[str, dict[str, Any]] = requirements.get(self.product_name(), {})
        for product, config in reqs.items():
            if product in (
                "nonlinear_Plk_grid",
                "nonlinear_Plk_interpolator",
                "nonlinear_Plk_gaussian_grid",
            ):
                chained = bool_or_list(config.get("chained", False))
                binned = bool_or_list(config.get("binned", False))
                if product == "nonlinear_Plk_interpolator" and True in binned:
                    raise LoggedError(self.log, "binned Plk interpolator not supported")
                for c, b in itertools.product(chained, binned):
                    self._must_provide[product].add((c, b))
            elif product == "snapshots":
                self._must_provide[product] = set()
            elif product == "eft_params_values_dict":
                self._must_provide[product] = set()
            elif product == "bird_component":
                self._must_provide[product] = set()
            else:
                raise LoggedError(
                    self.log,
                    "Unexpected requirement %s, this should not happen, "
                    "please contact the developers",
                    product,
                )
        self.mpi_debug("updated must provide %s", self._must_provide)

    def calculate(self, state, want_derived=True, **params_values_dict):
        tstart = perf_counter()
        if kernel_product := self.provider.get_result(self.kernel_product_name()):
            bird = kernel_product["bird"]
            birdlike_products = kernel_product["birdlike_products"]
            basis = self.basis
            products: dict = {}
            for product, keyset in self._must_provide.items():
                if product == "snapshots":
                    products[product] = bird.snapshots
                elif product == "eft_params_values_dict":
                    products[product] = {
                        p: params_values_dict.get(p, 0.0)
                        for p in basis.gaussian_params() + basis.non_gaussian_params()
                    }
                elif product == "bird_component":
                    ls, k, birdlike = kernel_product[product + "_product"]
                    component = basis.reduce_Plk(birdlike, params_values_dict)
                    products[product] = (ls, k, component)
                elif product == "nonlinear_Plk_gaussian_grid":
                    for key in keyset:
                        ls, k, birdlike = birdlike_products[key]
                        # TODO: requires
                        PG_table = basis.reduce_Plk_gaussian_table(
                            birdlike, params_values_dict
                        )
                        products[(product,) + key] = (ls, k, PG_table)
                elif product == "nonlinear_Plk_grid":
                    for key in keyset:
                        ls, k, birdlike = birdlike_products[key]
                        products[(product,) + key] = (
                            ls,
                            k,
                            basis.reduce_Plk(birdlike, params_values_dict).sum(),
                        )
                elif product == "nonlinear_Plk_interpolator":
                    for key in keyset:
                        ls, k, birdlike = birdlike_products[key]
                        if tmp := products.get(("nonlinear_Plk_grid",) + key):
                            _, _, Plk = tmp
                        else:
                            Plk = basis.reduce_Plk(birdlike, params_values_dict).sum()
                        fn = PlkInterpolator(ls, k, Plk)
                        chained, binned = key
                        products[(product, chained)] = fn
            state[self.product_name()] = products
        tend = perf_counter()
        self.mpi_debug("calculate %s: %fs", self.product_name(), tend - tstart)

    def get_can_provide(self):
        return [self.product_name()]

    def get_can_support_params(self):
        return self.basis.gaussian_params()

    def product_name(self) -> str:
        return leaf_product_name(self.tracer)

    def kernel_product_name(self) -> str:
        return leaf_kernel_product_name(self.tracer)
