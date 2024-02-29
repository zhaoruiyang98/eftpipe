from __future__ import annotations
import matryoshka.emulator as emu
import numpy as np
from cobaya.theory import Theory
from collections.abc import Mapping
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .typing import ndarrayf

_cosmo_latex = {
    "logA": R"\log(10^{10} A_\mathrm{s})",
    "ns": R"n_\mathrm{s}",
    "Omega_m": R"\Omega_m",
    "Omega_b": R"\Omega_b",
    "h": "h",
    "Neff": R"N_{\mathrm{eff}}",
    "w0": "w_0",
}


def _possible_proposal(p):
    return {
        "logA": 0.01,
        "ns": 0.05,
    }.get(p, 0.1)


def _possible_prior(p):
    if p == "logA":
        return {"min": 0.1, "max": 10}
    if p == "ns":
        return {"min": 0.1, "max": 2.0}
    return {"min": -100.0, "max": 100.0}


def extract_params_info(cosmo: MatryoshkaCosmo):
    default_params = cosmo.default_cosmo_params.copy()
    emulated_params = cosmo.emulated_params
    xscaler = cosmo.emu_growth.scalers[0]
    info = {}
    for p, xmin, xmax in zip(
        emulated_params, xscaler.min_val, xscaler.min_val + xscaler.diff  # type: ignore
    ):
        info[p] = {
            "prior": {"dist": "uniform", "min": float(xmin), "max": float(xmax)},
            "proposal": float((xmax - xmin) / 10.0),
            "ref": float(default_params[p]),
            "latex": _cosmo_latex[p],
        }
    for p in set(default_params) - set(emulated_params):
        info[p] = {
            "prior": _possible_prior(p),
            "proposal": _possible_proposal(p),
            "ref": float(default_params[p]),
            "latex": _cosmo_latex[p],
        }
    return info


def make_matryoshka_theory_class(name: str, version: str = "class_aemulus"):
    return type(name, (MatryoshkaTheoryBase,), {}, version=version)


class MatryoshkaCosmo(Mapping):
    def __init__(self, version: str = "class_aemulus", z: float = 0.0):
        if version != "class_aemulus":
            raise NotImplementedError
        self.version = version
        self.z = z
        self.emu_transfer = emu.Transfer(version=version)
        self.emu_growth = emu.Growth(version=version)
        self.emulated_params = ("Omega_m", "Omega_b", "h", "Neff", "w0")
        self.default_cosmo_params = {
            "logA": 3.0448,
            "ns": 0.96605,
            "Omega_m": 0.3158,
            "Omega_b": 0.049388982622703914,
            "h": 0.6732,
            "Neff": 3.046,
            "w0": -1.0,
        }
        self._norm = 1.45347445e15
        self._kh = np.hstack([[1e-6], self.emu_transfer.kbins])
        self._logkh = np.log(self._kh)

    def set_cosmo_params(self, params_values_dict):
        # default values are from Planck 2018
        cosmo: dict[str, float] = {}
        for p, default in self.default_cosmo_params.items():
            cosmo[p] = params_values_dict.get(p, default)
        cosmo["H0"] = cosmo["h"] * 100.0
        cosmo["As"] = np.exp(cosmo["logA"]) * 1e-10
        cosmo["Omega_cdm"] = cosmo["Omega_m"] - cosmo["Omega_b"]
        cosmo["omegach2"] = cosmo["Omega_cdm"] * cosmo["h"] ** 2.0
        cosmo["omegabh2"] = cosmo["Omega_b"] * cosmo["h"] ** 2.0
        self._cosmo_params = cosmo

    def calculate(self, **params_values_dict):
        self.set_cosmo_params(params_values_dict)
        cosmovec = np.array([self[p] for p in self.emulated_params])
        self._Tkh_grid = np.hstack([[1.0], self.emu_transfer.emu_predict(cosmovec)[0]])  # type: ignore
        self._Dz_grid = self.emu_growth.emu_predict(cosmovec)[0]  # type: ignore
        _Tkh_fn = CubicSpline(self.logkh(), self._Tkh_grid)
        self._Tkh_fn = lambda kh: _Tkh_fn(np.log(kh))
        self._Dz_fn = CubicSpline(self.emu_growth.zbins, self._Dz_grid)

    def kh(self):
        return self._kh

    def logkh(self):
        return self._logkh

    def Tkh(self, kh=None, z: float | None = None) -> ndarrayf:
        kh = kh if kh is not None else self.kh()
        z = z if z is not None else self.z
        return self._Tkh_fn(kh) * self.Dz(z)  # type: ignore

    def Dz(self, z: float | None = None) -> float:
        z = z if z is not None else self.z
        return self._Dz_fn(z)  # type: ignore

    def Pkh(self, kh=None, z: float | None = None) -> ndarrayf:
        kh = kh if kh is not None else self.kh()
        ns = self["ns"]
        return self._norm * self["As"] * kh**ns * self.Tkh(kh, z=z) ** 2.0

    def f(self, z: float) -> float:
        """scale independent growth rate"""
        return -(1.0 + z) / self.Dz(z) * self._Dz_fn.derivative()(z)  # type: ignore

    def comoving_radial_distance(self, z):
        """Comoving distance in Mpc/h"""
        return 299792.458 / 100 * quad(lambda z: 1.0 / self.efunc(z), 0, z)[0]

    def Hubble(self, z):
        """Hubble parameter in km/s/Mpc"""
        return self["H0"] * self.efunc(z)

    def efunc(self, z):
        Omega_m, w0 = self["Omega_m"], self["w0"]
        return np.sqrt(Omega_m * (1.0 + z) ** 3.0 + (1.0 - Omega_m) * (1.0 + z) ** (3.0 + 3.0 * w0))

    # mapping interface
    def __getitem__(self, key):
        return self._cosmo_params[key]

    def __iter__(self):
        return iter(self._cosmo_params)

    def __len__(self):
        return len(self._cosmo_params)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(version={self.version!r}, z={self.z})"


class MatryoshkaTheoryBase(Theory):
    matryoshka_version: ClassVar[str]

    def __init_subclass__(cls, version: str = "class_aemulus"):
        super().__init_subclass__()
        matryoshka = MatryoshkaCosmo(version=version)
        cls.matryoshka_version = version
        cls.params = extract_params_info(matryoshka)

    def initialize(self):
        super().initialize()
        self.matryoshka = MatryoshkaCosmo(version=self.matryoshka_version)

    def calculate(self, state, want_derived=True, **params_values_dict):
        super().calculate(state, want_derived, **params_values_dict)
        self.matryoshka.calculate(**params_values_dict)
        state["matryoshkacosmo"] = self.matryoshka

    def get_requirements(self):
        return list(self.matryoshka.default_cosmo_params.keys())

    def get_can_provide(self):
        return ["matryoshkacosmo"]
