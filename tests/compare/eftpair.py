from __future__ import annotations

import numpy as np
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Any, cast, Dict, Generator, Iterable, Literal
from typing_extensions import TypedDict, NotRequired
from numpy import ndarray as NDArray
from cobaya import get_model
from eftpipe.interface import CobayaCambInterface
from eftpipe.tools import PathContext
from eftpipe.tools import recursively_update_dict
from eftpipe.tools import RandomParams

CACHE_ROOT = Path(__file__).resolve().parent / "cache"
EFTPIPE_CACHE = CACHE_ROOT / "eftpipe"
PYBIRDDEV_CACHE = CACHE_ROOT / "pybird_dev"
COBAYA_PATH = Path(__file__).resolve().parent.parent.parent / "cobaya"
if not CACHE_ROOT.exists():
    CACHE_ROOT.mkdir()
if not EFTPIPE_CACHE.exists():
    EFTPIPE_CACHE.mkdir()
if not PYBIRDDEV_CACHE.exists():
    PYBIRDDEV_CACHE.mkdir()


def get_kdata(kmin=0.02, kmax=0.25) -> NDArray:
    data = np.loadtxt(COBAYA_PATH / "data" / "mock" / "LRG_NGC.txt")
    k = data[:, 0]
    mask = (k >= kmin) & (k <= kmax)
    return k[mask]


class IRresumConfig(TypedDict):
    optiresum: bool


class APconfig(TypedDict):
    Om_AP: float
    z_AP: float


class WindowConfig(TypedDict):
    window_fourier_file: str
    window_configspace_file: str
    window_st: NotRequired[bool]


class FiberConfig(TypedDict):
    ktrust: float
    fs: float
    Dfc: float


class BinningConfig(TypedDict):
    kout: NDArray


class Config(TypedDict):
    z: float
    km: float
    nd: float
    Nl: Literal[2, 3]

    with_IRresum: bool
    with_APeffect: bool
    with_window: bool
    with_fiber: bool
    with_binning: bool
    IRresum: NotRequired[IRresumConfig]
    APeffect: NotRequired[APconfig]
    window: NotRequired[WindowConfig]
    fiber: NotRequired[FiberConfig]
    binning: NotRequired[BinningConfig]


CONFIG: Config = {
    "z": 0.7,
    "km": 0.7,
    "nd": 7.91e-05,
    "Nl": 3,
    "with_IRresum": True,
    "with_APeffect": True,
    "with_window": True,
    "with_fiber": False,
    "with_binning": True,
    "IRresum": {"optiresum": False},
    "APeffect": {"Om_AP": 0.307115, "z_AP": 0.7},
    "window": {
        "window_fourier_file": str(EFTPIPE_CACHE / "LRG_NGC_GB_interp.npy"),
        "window_configspace_file": (
            str(COBAYA_PATH / "data" / "window" / "LRG_NGC_GB_interp.dat")
        ),
        "window_st": False,
    },
    "binning": {"kout": get_kdata()},
}


def config_to_info(config: Config):
    d = deepcopy(cast(Dict[str, Any], config))
    for item in ("Nl", "with_binning", "binning"):
        d.pop(item, None)
    info = {
        "eftpipe.eftlss": {
            "cache_dir_path": str(EFTPIPE_CACHE),
            "tracers": {
                "x": {
                    "prefix": "",
                    "provider": "camb",
                    "z": config["z"],
                    "km": config["km"],
                    "nd": config["nd"],
                    **d,
                }
            },
        }
    }
    info["camb"] = {"extra_args": {"num_massive_neutrinos": 1, "nnu": 3.046}}
    info = {"theory": info, "likelihood": {"one": None}}
    info["params"] = {
        "logA": {"prior": {"min": 0, "max": 10}, "drop": True},
        "As": {"value": "lambda logA: 1e-10*np.exp(logA)"},
        "ns": {"prior": {"min": 0, "max": 10}},
        "H0": {"prior": {"min": 0, "max": 100}},
        "ombh2": {"prior": {"min": 0.005, "max": 0.1}},
        "omch2": {"prior": {"min": 0.001, "max": 0.99}},
        "tau": {"prior": {"min": 0, "max": 10}},
        "mnu": {"prior": {"min": 0, "max": 10}},
        "b1": {"prior": {"min": 0, "max": 5}},
        "c2": {"prior": {"min": -10, "max": 10}},
        "b3": {"prior": {"min": -10, "max": 10}},
        "c4": {"prior": {"min": -10, "max": 10}},
        "b2": {"value": "lambda c2, c4: (c2 + c4) / np.sqrt(2.)"},
        "b4": {"value": "lambda c2, c4: (c2 - c4) / np.sqrt(2.)"},
        "cct": {"prior": {"min": -10, "max": 10}},
        "cr1": {"prior": {"min": -10, "max": 10}},
        "cr2": {"prior": {"min": -10, "max": 10}},
        "ce0": {"prior": {"min": -10, "max": 10}},
        "cemono": {"prior": {"min": -10, "max": 10}},
        "cequad": {"prior": {"min": -10, "max": 10}},
    }
    return info


class EFTPipeTh:
    def __init__(self, config: Config) -> None:
        Nl: int = config["Nl"]
        with_binning: bool = config["with_binning"]
        binning: BinningConfig = config.get("binning", {"kout": get_kdata()})
        info = config_to_info(config)
        model = get_model(info)  # type: ignore
        ls = [2 * i for i in range(Nl)]
        if with_binning:
            model.add_requirements(
                {
                    "nonlinear_Plk_grid": {
                        "x": {
                            "ls": ls,
                            "binned": with_binning,
                            **({"binning": binning} if with_binning else {}),
                        },
                    }
                }
            )
        else:
            model.add_requirements({"nonlinear_Plk_interpolator": {"x": {"ls": ls}}})
        self.ls = ls
        self.model = model
        self.with_binning = with_binning
        self.kout = binning["kout"]

    def Plk(self, params_dict: dict):
        self.model.logpost(params_dict)
        if self.with_binning:
            return self.model.provider.get_nonlinear_Plk_grid("x", binned=True)[-1]
        fn = self.model.provider.get_nonlinear_Plk_interpolator("x")
        return fn(self.ls, self.kout)


class PybirdTh:
    def __init__(self, config: Config):
        fourier_name = config.get("window", {"window_fourier_file": "test"})[
            "window_fourier_file"
        ]
        fourier_name = Path(fourier_name).name
        windowPk = str(PYBIRDDEV_CACHE / fourier_name)
        settings = dict(
            output="bPk",
            multipole=config["Nl"],
            xdata=get_kdata(),
            z=config["z"],
            km=config["km"],
            kr=config["km"],  # NOTE: the default of setting of pybird is kr != km
            nd=config["nd"],
            with_stoch=True,
            optiresum=config.get("IRresum", {"optiresum": False})["optiresum"],
            with_resum=config["with_IRresum"],
            with_bias=False,
            kmax=0.3,
            with_AP=config["with_APeffect"],
            z_AP=config.get("APeffect", {"z_AP": config["z"]})["z_AP"],
            Omega_m_AP=config.get("APeffect", {"Om_AP": 0.31})["Om_AP"],
            with_window=config["with_window"],
            windowPk=windowPk,
            windowCf=config.get("window", {"window_configspace_file": "test"})[
                "window_configspace_file"
            ],
            with_binning=config["with_binning"],
            with_fibercol=config["with_fiber"],
        )
        self.Nl = config["Nl"]
        self.z = config["z"]

        from pybird.pybird import Correlator

        with PathContext(PYBIRDDEV_CACHE):
            self.corr = Correlator()
            self.corr.set(settings)

    def set_bolzman_provider(self, provider):
        self.provider = CobayaCambInterface(provider, z=self.z)

    def Plk(self, params_dict: dict) -> NDArray:
        provider = self.provider
        k11 = np.logspace(-5, 0, 200)
        p11 = provider.Pkh(k11)
        H, DA, f = provider.H, provider.DA, provider.f
        dct = {
            "b1": params_dict["b1"],
            "b2": (params_dict["c2"] + params_dict["c4"]) / np.sqrt(2),
            "b3": params_dict["b3"],
            "b4": (params_dict["c2"] - params_dict["c4"]) / np.sqrt(2),
            "cct": params_dict["cct"],
            "cr1": params_dict["cr1"],
            "cr2": params_dict["cr2"],
            "ce0": params_dict["ce0"],
            "ce1": params_dict["cemono"],
            "ce2": params_dict["cequad"],
        }
        self.corr.compute(
            {"k11": k11, "P11": p11, "f": f, "DA": DA, "H": H,}
        )
        return self.corr.get(bias=dct)  # type: ignore


class EFTPair:
    def __init__(
        self,
        n: int,
        fixnames: Iterable | None = None,
        fixgroups: Iterable | None = None,
        seed: int | None = None,
        config: Config = CONFIG,
        override: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.n = n
        self.fixnames = fixnames
        self.fixgroups = fixgroups
        if override is None:
            override = {}
        self.override = override

    def __len__(self):
        return self.n

    def __iter__(self) -> Generator[tuple[NDArray, NDArray], None, None]:
        if self.n == 0:
            return
        # setup
        self.eftpipe_th = EFTPipeTh(self.config)
        self.pybird_th = PybirdTh(self.config)

        parent = Path(__file__).resolve().parent
        with open(parent / "default_params.yaml", "r") as f:
            ref = yaml.safe_load(f)
        recursively_update_dict(ref, self.override)
        with open(parent / "covmat.txt", "r") as f:
            header = f.readline().strip("\n").split(" ")[1:]
        covmat = np.loadtxt(parent / "covmat.txt")
        randp = RandomParams(ref, covmat, header, seed=self.seed)
        randp.fix(names=self.fixnames, groups=self.fixgroups)

        for _ in range(self.n):
            params_dict = randp.random()
            provider = self.eftpipe_th.model.theory["eftpipe.eftlss.x"].provider
            self.pybird_th.set_bolzman_provider(provider)
            yield (
                self.eftpipe_th.Plk(params_dict),
                self.pybird_th.Plk(params_dict),
            )

    def __repr__(self) -> str:
        return (
            f"EFTPair({self.n}, "
            f"fixnames={self.fixnames}, fixgroups={self.fixgroups})"
        )
