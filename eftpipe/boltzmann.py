"""
extract boltzmann products used by eftlss
"""

from __future__ import annotations
import importlib
import os
import numpy as np
from abc import abstractmethod
from typing import Any, cast, Protocol, TYPE_CHECKING
from scipy.interpolate import interp1d
from cobaya.log import HasLogger

if TYPE_CHECKING:
    from cobaya.theories.cosmo import BoltzmannBase
    from cobaya.theories.cosmo import PowerSpectrumInterpolator
    from cobaya.theory import Provider
    from .matryoshka import MatryoshkaCosmo
    from .typing import ndarrayf


class BoltzmannExtractor(Protocol):
    """Extracting boltzmann products required by eftlss

    Notes
    -----
    The motivation of this protocol is, although cobaya tries to provide a
    unified interface for different Boltzmann codes, usage in some situations
    may still be different. Fortunately, eftlss only relies on a small subset of
    the functionalities of boltzmann codes, so we create an intermediate layer.
    """

    def initialize(self, zeff: float, use_cb: bool, zextra: list[float], **kwargs) -> None:
        """initialize the extractor, invocked by eftlss in the ``initialize`` method

        Parameters
        ----------
        zeff : float
            The effective redshift to be used
        use_cb : bool
            if true, compute the linear power spectrum of cdm + baryon,
            otherwise, compute the linear power spectrum of cdm + baryon + neutrino
        zextra : list[float]
            extra redshifts, maybe useful to build Pk interpolator,
            since Pk_interpolator typically requires at least 4 redshifts.
        """
        pass

    def initialize_with_provider(self, provider: Provider) -> None:
        """initialize the extractor with provider, invocked by eftlss in the ``initialize_with_provider`` method

        Parameters
        ----------
        provider : Provider
            The provider object of cobaya
        """
        pass

    def get_requirements(self) -> dict[str, Any]:
        """Cosmological requirements raised by eftlss"""
        return {}

    def calculate(self, **params_values_dict):
        """do calculation"""
        pass

    @abstractmethod
    def Pkh(self, kh: ndarrayf) -> ndarrayf:
        """Linear power spectrum, accept k in Mpc/h unit, returns in (Mpc/h)^3 unit"""
        ...

    @abstractmethod
    def f(self) -> float:
        """Scale-independent linear growth rate"""
        ...

    @abstractmethod
    def DA(self) -> float:
        """Angular diameter distance in Mpc unit, divided by (c/H0)"""
        ...

    @abstractmethod
    def H(self) -> float:
        """dimensionless Hubble rate"""
        ...

    def h(self) -> float | None:
        """Hubble parameter H0 / 100"""
        return None

    def rdrag(self) -> float | None:
        """Sound horizon at baryon drag epoch in Mpc unit"""
        return None

    def fsigma8_z(self) -> float:
        R"""
        Structure growth rate :math:`f\sigma_8`, as defined in eq. 33 of
        `Planck 2015 results. XIII.
        Cosmological parameters <https://arxiv.org/pdf/1502.01589.pdf>`_.
        """
        return -1


class CobayaBoltzmannExtractor(HasLogger, BoltzmannExtractor):
    def __init__(self) -> None:
        self.set_logger()

    def initialize(
        self,
        zeff: float,
        use_cb: bool = False,
        zextra: list[float] = [],
        **kwargs,
    ) -> None:
        self.zeff = zeff
        self.use_cb = use_cb
        self.zextra = list(zextra)
        self.vars_pairs = ("delta_nonu", "delta_nonu") if use_cb else ("delta_tot", "delta_tot")

    def initialize_with_provider(self, provider: Provider) -> None:
        self.provider: BoltzmannBase = provider  # type: ignore

    def get_requirements(self) -> dict[str, Any]:
        raise NotImplementedError

    def Pkh(self, kh: ndarrayf) -> ndarrayf:
        # extra_kmin=1e-6 is sufficient
        fn: PowerSpectrumInterpolator = self.provider.get_Pk_interpolator(
            var_pair=self.vars_pairs, nonlinear=False, extrap_kmin=1e-6
        )
        h = self.h()
        pkh = fn.P(self.zeff, kh * h) * h**3
        return pkh

    def f(self) -> float:
        raise NotImplementedError

    def DA(self) -> float:
        return (
            float(self.provider.get_angular_diameter_distance(self.zeff))
            * (self.h() * 100)
            / 299792.458
        )

    def H(self) -> float:
        return float(self.provider.get_Hubble(self.zeff)) / (self.h() * 100)

    def h(self) -> float:
        return float(self.provider.get_Hubble(0.0)) / 100.0

    def rdrag(self) -> float:
        return float(self.provider.get_param("rdrag"))

    # FIXME: deal with cb
    def fsigma8_z(self) -> float:
        return float(self.provider.get_fsigma8(self.zeff))


class CobayaCambExtractor(CobayaBoltzmannExtractor):
    """BoltzmannInterface which uses Cobaya's Camb Provider as backend"""

    def get_requirements(self) -> dict[str, Any]:
        z = self.zeff
        extra_zs = [] if z == 0.0 else [0.0]
        requires = {
            "Pk_interpolator": {
                "nonlinear": False,
                "z": [z] + self.zextra,
                "k_max": 5,
                "vars_pairs": self.vars_pairs,
            },
            "Hubble": {"z": extra_zs + [z]},
            "angular_diameter_distance": {"z": [z]},
            "fsigma8": {"z": [z]},
            "sigma8_z": {"z": [z]},
            "rdrag": None,
        }
        return requires

    # FIXME: I'm not sure if it is correct ...
    def f(self) -> float:
        return self.get_fsigma8(self.zeff) / self.get_sigma8_z(self.zeff)

    def get_fsigma8(self, z: float) -> float:
        return float(self.provider.get_fsigma8(z))

    def get_sigma8_z(self, z: float) -> float:
        return float(self.provider.get_sigma8_z(z))


class CobayaClassyExtractor(CobayaBoltzmannExtractor):
    """BoltzmannInterface which uses Cobaya's Classy Provider as backend"""

    def initialize_with_provider(self, provider: Provider) -> None:
        super().initialize_with_provider(provider)
        self.name = None
        for k in provider.model.theory.keys():
            # XXX: name collision
            if "classy" in k:
                self.name = k
                break
        if self.name is None:
            raise ValueError("classy or classynu not in theory")
        self.name = cast(str, self.name)

    def get_requirements(self) -> dict[str, Any]:
        z = self.zeff
        extra_zs = [] if z == 0.0 else [0.0]
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
                "vars_pairs": self.vars_pairs,
            },
            "Hubble": {"z": extra_zs + [z]},
            "angular_diameter_distance": {"z": [z]},
            "fsigma8": {"z": [z]},
            "rdrag": None,
        }
        return requires

    def f(self) -> float:
        from classy import CosmoSevereError  # type: ignore

        try:
            return self.provider.model.theory[  # type: ignore
                self.name
            ].classy.scale_independent_growth_factor_f(self.zeff)
        except CosmoSevereError:
            self.mpi_warning(
                "classy failed to compute f, current cosmological parameters: %r",
                self.provider.model.theory[self.name]._states[0]["params"],  # type: ignore
            )
            raise


class LinearPowerFile(HasLogger, BoltzmannExtractor):
    def __init__(self, path: str | os.PathLike, gz: float = 1, prefix: str = "") -> None:
        self.set_logger()
        k, pk = np.loadtxt(path, unpack=True)
        self.mpi_info("loading linear power spectrum from %s", path)
        self.mpi_info("growth factor assumed to be %f", gz)
        pk *= gz**2
        self.mpi_info(
            "require parameter %s, %s and %s",
            prefix + "f",
            prefix + "alperp",
            prefix + "alpara",
        )
        self.prefix = prefix
        if k[0] > 1e-5:
            print(k[1], k[0])
            ns = (np.log(pk[1]) - np.log(pk[0])) / (np.log(k[1]) - np.log(k[0]))
            lowk = np.geomspace(1e-5, k[0], 100, endpoint=False)
            lowpk = pk[0] * (lowk / k[0]) ** ns
            k = np.hstack((lowk, k))
            pk = np.hstack((lowpk, pk))
            self.mpi_info("extrapolating linear power spectrum to k=1e-5, ns=%f", ns)
        fn = interp1d(np.log(k), np.log(pk), kind="cubic")
        self.plin = lambda k: np.exp(fn(np.log(k)))
        # klim = np.logspace(-5, 1, 200)
        # plim = self.plin(klim)
        # from matplotlib import pyplot as plt
        # plt.loglog(klim, plim)
        # plt.show()
        # exit()
        self._returned_DA = False
        self._returned_H = False

    def initialize(self, zeff: float, use_cb: bool, zextra: list[float], **kwargs) -> None:
        if use_cb:
            self.mpi_warning("use_cb is ignored for LinearPowerFile")

    def initialize_with_provider(self, provider: Provider) -> None:
        self.provider = provider

    def get_requirements(self) -> dict[str, Any]:
        return {
            self.prefix + "f": None,
            self.prefix + "alperp": None,
            self.prefix + "alpara": None,
        }

    def Pkh(self, kh: ndarrayf) -> ndarrayf:
        return self.plin(kh)

    def f(self) -> float:
        return self.provider.get_param(self.prefix + "f")  # type: ignore

    def DA(self) -> float:
        if self._returned_DA:
            return self.provider.get_param(self.prefix + "alperp")  # type: ignore
        self._returned_DA = True
        return 1

    def H(self) -> float:
        if self._returned_H:
            return 1 / self.provider.get_param(self.prefix + "alpara")  # type: ignore
        self._returned_H = True
        return 1

    def h(self) -> float:
        return 1

    def rdrag(self) -> float:
        return 1


class MatryoshkaBoltzmannExtractor(BoltzmannExtractor):
    def initialize(self, zeff: float, use_cb: bool, zextra: list[float], **kwargs) -> None:
        self.zeff = zeff
        if use_cb:
            raise NotImplementedError

    def initialize_with_provider(self, provider: Provider) -> None:
        self.provider = provider

    def get_requirements(self):
        return {"matryoshkacosmo": None}

    def _get_cosmo(self) -> MatryoshkaCosmo:
        return self.provider.get_result("matryoshkacosmo")

    def Pkh(self, kh: ndarrayf) -> ndarrayf:
        return self._get_cosmo().Pkh(kh, z=self.zeff)

    def f(self) -> float:
        return self._get_cosmo().f(self.zeff)

    def DA(self) -> float:
        coeff = 100 / 299792.458 / (1 + self.zeff)
        return self._get_cosmo().comoving_radial_distance(z=self.zeff) * coeff

    def H(self) -> float:
        cosmo = self._get_cosmo()
        return cosmo.Hubble(self.zeff) / cosmo["H0"]

    def h(self) -> float:
        return self._get_cosmo()["h"]


def find_boltzmann_extractor(name, kwargs: dict[str, Any]) -> BoltzmannExtractor:
    if not isinstance(name, str):
        # instance
        return name
    if name == "camb":
        return CobayaCambExtractor()
    if name in ("classy", "classynu"):
        return CobayaClassyExtractor()
    if name == "matryoshka":
        return MatryoshkaBoltzmannExtractor()
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)


if TYPE_CHECKING:

    def assert_protocol(x: BoltzmannExtractor):
        pass

    assert_protocol(CobayaClassyExtractor())
    assert_protocol(CobayaCambExtractor())
    assert_protocol(LinearPowerFile(""))
