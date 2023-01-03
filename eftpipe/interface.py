"""
boltzmann interface used by eftlss
"""
from __future__ import annotations
import importlib
import numpy as np
from copy import deepcopy
from typing import Any, cast, Protocol, TYPE_CHECKING
from cobaya.log import HasLogger

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from cobaya.theories.cosmo import BoltzmannBase
    from cobaya.theories.cosmo import PowerSpectrumInterpolator
    from cobaya.theory import Provider


class BoltzmannInterface(Protocol):
    """Protocol for the Boltzmann interface used by eftlss

    Notes
    -----
    The motivation of this protocol is, although cobaya tries to provide a
    unified interface for different Boltzmann codes, usage in some situations
    may still be different. Fortunately, eftlss only relies on a small subset of
    the functionalities of boltzmann codes, so we create an intermediate layer
    for specialization.
    """

    def initialize(self, zeff: float, use_cb: bool, zextra: list[float]) -> None:
        """This function would be invocked by eftlss in ``initialize`` method

        Parameters
        ----------
        zeff : float
            The effective redshift to be used
        use_cb : bool
            if true, compute the linear power spectrum of cdm + baryon,
            otherwise, compute the linear power spectrum of cdm + baryon + neutrino
        zextra : list[float]
            extra redshifts, maybe useful to build Pk interpolator,
            since Pk_interpolator requires at least 4 redshift
        """
        ...

    def initialize_with_provider(self, provider: Provider) -> None:
        """This function would be invocked by eftlss in ``initialize_with_provider`` method

        Parameters
        ----------
        provider : Provider
            The provider object of cobaya
        """

    def get_requirements(self) -> dict[str, Any]:
        """Cosmological requirements of this interface when used by eftlss"""
        ...

    def updated(self) -> bool:
        """Whether the Boltzmann code has been updated, compared to the last time it was invoked"""
        ...

    def Pkh(self, kh: NDArray[np.floating]) -> NDArray[np.floating]:
        """Linear power spectrum, accept k in Mpc/h unit, returns in (Mpc/h)^3 unit"""
        ...

    def f(self) -> float:
        """Scale-independent linear growth rate"""
        ...

    def h(self) -> float:
        """Hubble parameter H0 / 100"""
        ...

    def DA(self) -> float:
        """Angular diameter distance in Mpc/h unit"""
        ...

    def H(self) -> float:
        """Hubble rate in km/s/Mpc/h unit"""
        ...

    def rdrag(self) -> float:
        """Sound horizon at baryon drag epoch in Mpc unit"""
        ...

    def fsigma8_z(self) -> float:
        R"""
        Structure growth rate :math:`f\sigma_8`, as defined in eq. 33 of
        `Planck 2015 results. XIII.
        Cosmological parameters <https://arxiv.org/pdf/1502.01589.pdf>`_.
        """
        ...


class InternalBoltzmannInterface(HasLogger):
    def __init__(self) -> None:
        self.set_logger()

    def initialize(
        self,
        zeff: float,
        use_cb: bool = False,
        zextra: list[float] = [],
    ) -> None:
        self.zeff = zeff
        self.use_cb = use_cb
        self.zextra = list(zextra)
        self.vars_pairs = (
            ("delta_nonu", "delta_nonu") if use_cb else ("delta_tot", "delta_tot")
        )

    def initialize_with_provider(self, provider: Provider) -> None:
        self.provider: BoltzmannBase = provider  # type: ignore
        self.cosmo_params_dict = {}

    def get_requirements(self) -> dict[str, Any]:
        raise NotImplementedError

    def updated(self) -> bool:
        """
        Notes
        -----
        This is a hack to cobaya's classy and camb modules
        """
        raise NotImplementedError

    def Pkh(self, kh: NDArray[np.floating]) -> NDArray[np.floating]:
        # extra_kmin=1e-6 is sufficient
        fn: PowerSpectrumInterpolator = self.provider.get_Pk_interpolator(
            var_pair=self.vars_pairs, nonlinear=False, extrap_kmin=1e-6
        )
        h = self.h()
        pkh = fn.P(self.zeff, kh * h) * h**3
        return pkh

    def f(self) -> float:
        raise NotImplementedError

    def h(self) -> float:
        return float(self.provider.get_Hubble(0.0)) / 100.0

    def DA(self) -> float:
        return (
            float(self.provider.get_angular_diameter_distance(self.zeff))
            * (self.h() * 100)
            / 299792.458
        )

    def H(self) -> float:
        return float(self.provider.get_Hubble(self.zeff)) / (self.h() * 100)

    def rdrag(self) -> float:
        return float(self.provider.get_param("rdrag"))

    def fsigma8_z(self) -> float:
        return float(self.provider.get_fsigma8(self.zeff))


class CobayaCambInterface(InternalBoltzmannInterface):
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

    # HACK
    def updated(self) -> bool:
        flag = True
        provider = cast(Any, self.provider)
        transfer = provider.model.theory["camb.transfers"]
        camb = provider.model.theory["camb"]
        if len(transfer._states) != 0 and len(camb._states) != 0:
            cosmo_params_dict = deepcopy(transfer._states[0]["params"])
            cosmo_params_dict.update(camb._states[0]["params"])
            if cosmo_params_dict == self.cosmo_params_dict:
                flag = False
            else:
                self.cosmo_params_dict = cosmo_params_dict
        return flag

    # XXX: warning: f used here is effective f, which is different from that used in classy
    # possiblely use Omega_m(z)**0.545 instead
    def f(self) -> float:
        return self.get_fsigma8(self.zeff) / self.get_sigma8_z(self.zeff)

    def get_fsigma8(self, z: float) -> float:
        return float(self.provider.get_fsigma8(z))

    def get_sigma8_z(self, z: float) -> float:
        return float(self.provider.get_sigma8_z(z))


class CobayaClassyInterface(InternalBoltzmannInterface):
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

    # HACK
    def updated(self) -> bool:
        flag = True
        classy = self.provider.model.theory[self.name]  # type: ignore
        if len(classy._states) != 0:
            cosmo_params_dict = deepcopy(classy._states[0]["params"])
            if cosmo_params_dict == self.cosmo_params_dict:
                flag = False
            else:
                self.cosmo_params_dict = cosmo_params_dict
        return flag

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


def find_boltzmann_interface(name: str, kwargs: dict[str, Any]) -> BoltzmannInterface:
    if name == "camb":
        ret = CobayaCambInterface()
    elif name in ("classy", "classynu"):
        ret = CobayaClassyInterface()
    else:
        module_name, class_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        ret = getattr(module, class_name)(**kwargs)
    return ret


if TYPE_CHECKING:

    def assert_protocol(x: BoltzmannInterface):
        pass

    assert_protocol(CobayaClassyInterface())
    assert_protocol(CobayaCambInterface())
