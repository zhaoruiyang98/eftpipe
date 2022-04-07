import pytest
import numpy as np
from numpy import ndarray as NDArray
import yaml
from pathlib import Path
from typing import (
    Generator,
    Tuple,
)
from eftpipe.interface import CambProvider
from eftpipe.lssdata import PklData
from eftpipe.theory import EFTTheory
from eftpipe.tools import PathContext
from eftpipe.tools import RandomParams
try:
    from pybird_dev.pybird import Correlator
except ImportError:
    pass

CACHE_ROOT = Path(__file__).resolve().parent / 'cache'
EFTPIPE_CACHE = CACHE_ROOT / 'eftpipe'
PYBIRDDEV_CACHE = CACHE_ROOT / 'pybird_dev'
COBAYA_PATH = Path(__file__).resolve().parent.parent.parent / 'cobaya'
if not CACHE_ROOT.exists():
    CACHE_ROOT.mkdir()
if not EFTPIPE_CACHE.exists():
    EFTPIPE_CACHE.mkdir()
if not PYBIRDDEV_CACHE.exists():
    PYBIRDDEV_CACHE.mkdir()


def get_kdata(kmin=0.02, kmax=0.25):
    data = PklData.loadtxt(
        path=(COBAYA_PATH / 'data' / 'mock' / 'LRG_NGC.txt'),
        log=False,
    )
    data.set_mask(ls=[0, 2, 4], kmin=kmin, kmax=kmax)
    return data.kdata


HYPERPARAMS = {
    'km': 0.7, 'knl': 0.7, 'nd': 7.91e-05,
}

EFTPIPE_SETTINGS = {
    'z': 0.7,
    'km': HYPERPARAMS['km'],
    'nd': HYPERPARAMS['nd'],
    'cache_dir_path': EFTPIPE_CACHE,
    'optiresum': False,
    'Nl': 3,
    'with_IRresum': True,
    'with_APeffect': True,
    'with_window': True,
    'with_fiber': False,
    'with_binning': True,
    'config_settings': {
        "APeffect": {"Om_AP": 0.307115, "z_AP": 0.7},
        "window": {
            'window_fourier_file': EFTPIPE_CACHE / 'LRG_NGC_GB_interp.npy',
            'window_configspace_file': \
            COBAYA_PATH / 'data' / 'window' / 'LRG_NGC_GB_interp.dat',
        },
        "binning": {
            "kout": get_kdata(),
            "binning": True,
        }
    }
}

# when modifying these settings, be carefull because pybird_dev has some
# unexpected behaviour, e.g., if multipole is set to 2, cr2 should not be passed
# to Correlator.set
PYBIRDDEV_SETTINGS = {
    'output': 'bPk',
    'multipole': EFTPIPE_SETTINGS['Nl'],
    'xdata': EFTPIPE_SETTINGS['config_settings']['binning']['kout'],
    'z': EFTPIPE_SETTINGS['z'],
    'km': HYPERPARAMS['km'],
    'nd': HYPERPARAMS['nd'],
    'with_stoch': True,
    'optiresum': EFTPIPE_SETTINGS['optiresum'],
    'with_resum': True,
    'with_bias': False,
    'kmax': 0.3,
    'with_AP': True,
    'z_AP': EFTPIPE_SETTINGS['config_settings']['APeffect']['z_AP'],
    'Omega_m_AP': EFTPIPE_SETTINGS['config_settings']['APeffect']['Om_AP'],
    'with_window': True,
    'windowPk': str(PYBIRDDEV_CACHE / 'LRG_NGC_GB_interp'),
    'windowCf': str(
        COBAYA_PATH / 'data' / 'window' / 'LRG_NGC_GB_interp.dat'
    ),
    'with_binning': EFTPIPE_SETTINGS['config_settings']['binning']['binning'],
    'with_fibercol': False
}


class CambProviderNoCache(CambProvider):
    # override
    def cosmo_updated(self) -> bool:
        return True


class EFTPipeTh:
    def __init__(self, settings):
        self.theory = EFTTheory(**settings)  # type: ignore

    def set_boltzmann_provider(self, provider):
        self.theory.set_boltzmann_provider(provider)

    def theory_vector(self, params_dict):
        bs = [
            params_dict['b1'],
            (params_dict['c2'] + params_dict['c4']) / np.sqrt(2),
            params_dict['b3'],
            (params_dict['c2'] - params_dict['c4']) / np.sqrt(2),
            params_dict['cct'],
            params_dict['cr1'],
            params_dict['cr2']
        ]
        es = [
            params_dict['ce0'],
            params_dict['cemono'],
            params_dict['cequad'],
        ]
        return self.theory.theory_vector(bs, es=es)


class PybirdDevTh:
    def __init__(self, settings) -> None:
        with PathContext(PYBIRDDEV_CACHE):
            self.corr = Correlator()
            self.corr.set(settings)
        self.z = settings['z']

    def set_bolzman_provider(self, provider):
        self.bolzmann_provider = provider

    def theory_vector(self, params_dict):
        provider = self.bolzmann_provider
        k11 = np.logspace(-4, 0, 200)
        p11 = provider.interp_pkh(k11)
        H, DA, f = (
            provider.get_H(self.z),
            provider.get_DA(self.z),
            provider.get_f(self.z),
        )
        dct = {
            'b1': params_dict['b1'],
            'b2': (params_dict['c2'] + params_dict['c4']) / np.sqrt(2),
            'b3': params_dict['b3'],
            'b4': (params_dict['c2'] - params_dict['c4']) / np.sqrt(2),
            'cct': params_dict['cct'],
            'cr1': params_dict['cr1'],
            'cr2': params_dict['cr2'],
            'ce0': params_dict['ce0'],
            'ce1': params_dict['cemono'],
            'ce2': params_dict['cequad'],
        }
        self.corr.compute({
            'k11': k11,
            'P11': p11,
            'f': f, 'DA': DA, 'H': H,
        })
        return self.corr.get(bias=dct).reshape(-1)  # type: ignore


class EFTPair:
    def __init__(
        self,
        n,
        fixnames=None,
        fixgroups=None,
        seed=None,
        eftpipe_settings=EFTPIPE_SETTINGS,
        pybird_settings=PYBIRDDEV_SETTINGS,
        override=None,
    ) -> None:
        self.eftpipe_settings = eftpipe_settings
        self.pybird_settings = pybird_settings
        self.seed = seed
        self.n = n
        self.fixnames = fixnames
        self.fixgroups = fixgroups
        self.z = eftpipe_settings['z']
        if override is None:
            override = {}
        self.override = override

    def __len__(self):
        return self.n

    def __iter__(self) -> Generator[Tuple[NDArray, NDArray], None, None]:
        # setup
        self.eftpipe_th = EFTPipeTh(self.eftpipe_settings)
        self.pybird_th = PybirdDevTh(self.pybird_settings)

        parent = Path(__file__).resolve().parent
        with open(parent / "default_params.yaml", 'r') as f:
            ref = yaml.safe_load(f)
        if self.override:
            for k, v in ref.items():
                if isinstance(v, dict):
                    for kk in v.keys():
                        if kk in self.override:
                            ref[k][kk] = self.override[kk]
                else:
                    if k in self.override:
                        ref[k] = self.override[k]
        with open(parent / "covmat.txt", 'r') as f:
            header = f.readline().strip('\n').split(' ')[1:]
        covmat = np.loadtxt(parent / 'covmat.txt')
        randp = RandomParams(ref, covmat, header, seed=self.seed)
        randp.fix(names=self.fixnames, groups=self.fixgroups)
        cosmonames = ['ombh2', 'omch2', 'H0', 'logA', 'ns', 'mnu', 'tau']

        for _ in range(self.n):
            params_dict = randp.random()
            cosmo_dict = {
                k: v for k, v in params_dict.items() if k in cosmonames}
            provider = CambProviderNoCache(z=self.z, **cosmo_dict)
            self.eftpipe_th.set_boltzmann_provider(provider)
            self.pybird_th.set_bolzman_provider(provider)
            yield (self.eftpipe_th.theory_vector(params_dict),
                   self.pybird_th.theory_vector(params_dict))

    def __repr__(self) -> str:
        return f"EFTPair({self.n}, "\
               f"fixnames={self.fixnames}, fixgroups={self.fixgroups})"


def has_pybird_dev() -> bool:
    try:
        Correlator # type: ignore
    except NameError:
        return False
    else:
        return True


N = 5
it = (_ for _ in EFTPair(
    N, fixgroups=('stochastic',), override={'ce0': 0, 'cemono': 0, 'cequad': 0}
))
@pytest.mark.skipif(not has_pybird_dev(), reason='should install pybird_dev')
@pytest.mark.diffbird
@pytest.mark.parametrize("i", range(N))
def test_EFTtheory_vs_pybirddev(compare_ndarrays, i):
    eftpipe_th, pybird_th = next(it)
    eftpipe_th = eftpipe_th.reshape((3, -1))
    pybird_th = pybird_th.reshape((3, -1))
    eftpipe_dict = {k: v for k, v in zip(('P0', 'P2', 'P4'), eftpipe_th)}
    pybird_dict = {k: v for k, v in zip(('P0', 'P2', 'P4'), pybird_th)}
    compare_ndarrays(
        pybird_dict,
        eftpipe_dict,
        tolerances={
            'P0': {'atol': 0., 'rtol': 1e-4},
            'P2': {'atol': 0., 'rtol': 1e-4},
            'P4': {'atol': 0., 'rtol': 1e-4}
        },
    )
