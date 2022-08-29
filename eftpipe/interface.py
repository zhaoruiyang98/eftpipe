"""
Thin wrapper for boltzmann provider
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from cobaya.theories.cosmo import BoltzmannBase
    from cobaya.theories.cosmo import PowerSpectrumInterpolator


class BoltzmannInterface:
    """Base BoltzmannInterface

    Parameters
    ----------
    provider : Provider
        Boltzmann provider
    z : float
        redshift
    use_cb: bool
        compute the linear power spectrum of cdm + baryon,
        by default False and compute the total matter power spectrum
    """

    def __init__(self, provider: Any, z: float, use_cb: bool = False) -> None:
        self.provider: BoltzmannBase = provider
        self.z = z
        self.var_pair = (
            ("delta_nonu", "delta_nonu") if use_cb else ("delta_tot", "delta_tot")
        )
        self.cosmo_params_dct = {}

    def Pkh(self, kh: NDArray) -> NDArray:
        # extra_kmin=1e-6 is sufficient
        fn: PowerSpectrumInterpolator = self.provider.get_Pk_interpolator(
            self.var_pair, nonlinear=False, extrap_kmin=1e-6
        )
        h = self.h
        pkh = fn.P(self.z, kh * h) * h ** 3
        return pkh

    @property
    def h(self) -> float:
        return float(self.provider.get_Hubble(0.0)) / 100.0

    @property
    def DA(self) -> float:
        return (
            float(self.provider.get_angular_diameter_distance(self.z))
            * (self.h * 100)
            / 299792.458
        )

    @property
    def H(self) -> float:
        return float(self.provider.get_Hubble(self.z)) / (self.h * 100)

    @property
    def f(self) -> float:
        raise NotImplementedError

    @property
    def rdrag(self) -> float:
        return float(self.provider.get_param("rdrag"))

    @property
    def fsigma8_z(self) -> float:
        return float(self.provider.get_fsigma8(self.z))

    def updated(self) -> bool:
        """compare cosmological parameters with the ones at last invocation

        Notes
        -----
        This is a hack to cobaya's classy and camb modules
        """
        raise NotImplementedError


class CobayaCambInterface(BoltzmannInterface):
    """BoltzmannInterface which uses Cobaya's Camb Provider as backend

    Parameters
    ----------
    provider: Provider
        cobaya's camb provider
    z: float
        redshift
    use_cb: bool
        compute the linear power spectrum of cdm + baryon,
        by default False and compute the total matter power spectrum
    """

    # XXX: warning: f computed here is different from that used in classy
    @property
    def f(self) -> float:
        return self.get_fsigma8(self.z) / self.get_sigma8_z(self.z)

    def get_fsigma8(self, z) -> float:
        return float(self.provider.get_fsigma8(z))

    def get_sigma8_z(self, z) -> float:
        return float(self.provider.get_sigma8_z(z))

    # HACK
    def updated(self) -> bool:
        flag = True
        provider = cast(Any, self.provider)
        # XXX: support other names (this will not be a problem for internal camb)
        transfer = provider.model.theory["camb.transfers"]
        camb = provider.model.theory["camb"]
        if len(transfer._states) != 0 and len(camb._states) != 0:
            cosmo_params_dct = deepcopy(transfer._states[0]["params"])
            cosmo_params_dct.update(camb._states[0]["params"])
            if cosmo_params_dct == self.cosmo_params_dct:
                flag = False
            else:
                self.cosmo_params_dct = cosmo_params_dct
        return flag


class CobayaClassyInterface(BoltzmannInterface):
    """BoltzmannInterface which uses Cobaya's Classy Provider as backend

    Parameters
    ----------
    provider: Provider
        cobaya's classy provider
    z: float
        redshift
    use_cb: bool
        compute the linear power spectrum of cdm + baryon,
        by default False and compute the total matter power spectrum
    """

    def __init__(self, provider: Any, z: float, use_cb: bool = False) -> None:
        self.name = None
        for k in provider.model.theory.keys():
            # XXX: support other names (this will not be a problem for internal classy)
            if "classy" in k:
                self.name = k
                break
        if self.name is None:
            raise ValueError("classy or classynu not in theory")
        self.name = cast(str, self.name)
        super().__init__(provider, z=z, use_cb=use_cb)

    @property
    def f(self) -> float:
        return self.provider.model.theory[  # type: ignore
            self.name
        ].classy.scale_independent_growth_factor_f(self.z)

    # HACK
    def updated(self) -> bool:
        flag = True
        classy = self.provider.model.theory[self.name]  # type: ignore
        if len(classy._states) != 0:
            cosmo_params_dct = deepcopy(classy._states[0]["params"])
            if cosmo_params_dct == self.cosmo_params_dct:
                flag = False
            else:
                self.cosmo_params_dct = cosmo_params_dct
        return flag
