import camb
import numpy as np
from copy import deepcopy
from numpy import ndarray as NDArray
from scipy.interpolate import interp1d
from cobaya.theory import Provider
from camb import CAMBparams
from typing import Optional


class CambProvider:
    """BoltzmannProvider which uses CAMB as backend

    Parameters
    ----------
    pars: CAMBparams or None
        if given, use it to compute results, default None
    ombh2: float
        Omega_baryon h^2, default 0.02220129
    omch2: float
        Omega_cdm h^2, default 0.12011
    H0: float
        Hubble parameter in km/s/Mpc units, default 67.6
    logA: float
        log comoving curvature power at k=pivot_scalar, default 3.04
    ns: float
        scalar spectral index, default 0.97
    mnu: float
        sum of neutrino masses (in eV), default 0.06
    tau: float
        optical depth, default 0.0543
    z: float
        redshift, default 0

    Notes
    -----
    It has been tested that CambProvider and CobayaCambProvider are generally consistent
    1. rdrag, Hubble, DA: relative difference <= 1e-6
    2. fsigma8, sigma8: relative difference <= 1e-4
    3. for kh=np.logspace(-4, 0, 200), pkh: relative difference <= 2e-4
    """

    def __init__(
        self,
        pars: Optional[CAMBparams] = None,
        ombh2: float = 0.02220129,
        omch2: float = 0.12011,
        H0: float = 67.6,
        logA: float = 3.04,
        ns: float = 0.97,
        mnu: float = 0.06,
        tau: float = 0.0543,
        z: float = 0.0
    ) -> None:
        self.z = z
        if pars is not None:
            results = camb.get_results(pars)
        else:
            mypars = CAMBparams(WantTransfer=True, WantCls=False)
            mypars.set_cosmology(
                ombh2=ombh2, omch2=omch2,
                H0=H0, tau=tau, mnu=mnu,
            )
            mypars.InitPower.set_params(
                ns=ns, As=1e-10 * np.exp(logA))  # type: ignore
            mypars.set_matter_power(redshifts=(z,), nonlinear=False)
            results = camb.get_results(mypars)
        self.results = results
        kinterp, zs, pkhinterp = self.results.get_linear_matter_power_spectrum(
            nonlinear=False)
        index = list(zs).index(self.z)
        pkhinterp = pkhinterp[index, :]
        self._interp_pkh = interp1d(kinterp, pkhinterp, kind='cubic')

    def interp_pkh(self, kh: NDArray) -> NDArray:
        return self._interp_pkh(kh)

    def get_h0(self) -> float:
        return float(self.results.hubble_parameter(0) / 100.)

    def get_Hubble(self, z: float) -> float:
        return float(self.results.hubble_parameter(z))

    def get_angular_diameter_distance(self, z: float) -> float:
        return float(self.results.angular_diameter_distance(z))

    def get_fsigma8(self, z: float) -> float:
        return float(self.results.get_fsigma8())

    def get_sigma8_z(self, z: float) -> float:
        return float(self.results.get_sigma8())

    def get_rdrag(self) -> float:
        return self.results.get_derived_params()['rdrag']

    def cosmo_updated(self):
        return False


class CobayaCambProvider:
    """BoltzmannProvider which uses Cobaya's Camb Provider as backend

    Parameters
    ----------
    provider: Provider
        cobaya's camb provider
    z: float
        redshift
    """

    def __init__(self, provider: Provider, z: float) -> None:
        self.provider = provider
        self.z = z
        self.cosmo_params_dct = {}

    def interp_pkh(self, kh: NDArray) -> NDArray:
        kinterp, zs, pkinterp = self.provider.get_Pk_grid(nonlinear=False)
        h = float(self.get_h0())
        khinterp = kinterp / h
        index = list(zs).index(self.z)
        pkh = pkinterp[index, :] * h**3
        khinterp = np.insert(khinterp, 0, 0.0)
        pkh = np.insert(pkh, 0, 0.0)
        interp_pkh = interp1d(khinterp, pkh, kind='cubic')
        return interp_pkh(kh)

    def get_h0(self):
        return float(self.provider.get_Hubble(0.)) / 100.

    def get_Hubble(self, z):
        return float(self.provider.get_Hubble(z))

    def get_angular_diameter_distance(self, z):
        return float(self.provider.get_angular_diameter_distance(z))

    def get_fsigma8(self, z):
        return float(self.provider.get_fsigma8(z))

    def get_sigma8_z(self, z):
        return float(self.provider.get_sigma8_z(z))

    def get_rdrag(self):
        return float(self.provider.get_param('rdrag'))  # type: ignore

    def cosmo_updated(self):
        flag = True
        # TODO: support other names
        transfer = self.provider.model.theory['camb.transfers']
        camb = self.provider.model.theory['camb']
        if len(transfer._states) != 0 and len(camb._states) != 0:
            cosmo_params_dct = deepcopy(transfer._states[0]['params'])
            cosmo_params_dct.update(camb._states[0]['params'])
            if cosmo_params_dct == self.cosmo_params_dct:
                flag = False
            else:
                self.cosmo_params_dct = cosmo_params_dct
        return flag
