import numpy as np
from dataclasses import dataclass
from numpy import ndarray as NDArray
from scipy.interpolate import interp1d
from scipy.special import legendre
from pathlib import Path
from typing import (
    Iterable,
    List,
    Optional,
    Any,
    Callable,
)
from eftpipe.pybird import pybird
# Local
from eftpipe.typing import (
    ProjectionConfig,
    BoltzmannProvider,
)


def chain_coeff(l: int) -> float:
    r"""compute A_\ell coeff for chained power spectrum

    Parameters
    ----------
    l : int

    Returns
    -------
    float

    Notes
    -----
        .. math::
            \frac{(2\ell+1)\mathcal{L}_{\ell}(0)}{(2\ell+5)\mathcal{L}_{\ell+2}(0)}
    """
    return ((2 * l + 1) * legendre(l)(0)) / ((2 * l + 5) * legendre(l + 2)(0))


class BirdPlus(pybird.Bird):
    # override
    def setreducePslb(
        self,
        bsA: Iterable[float],
        bsB: Optional[Iterable[float]] = None
    ) -> None:
        """apply counter terms and bind fullPs to self

        Parameters
        ----------
        bsA : Iterable[float]
            b_1, b_2, b_3, b_4, c_{ct}/k_{nl}^2, c_{r,1}/k_{m}^2, c{r,2}/k_{m}^2
        bsB : Optional[Iterable[float]], optional
            the same as bsA, but for tracer A, by default None
            leave it default will compute auto power spectrum
        """
        b1A, b2A, b3A, b4A, b5A, b6A, b7A = bsA
        if bsB is None:
            bsB = bsA
        b1B, b2B, b3B, b4B, b5B, b6B, b7B = bsB
        f = self.f

        b11AB = np.array([b1A*b1B, (b1A + b1B) * f, f**2])
        bctAB = np.array([
            b1A * b5B + b1B * b5A, b1B * b6A + b1A * b6B,
            b1B * b7A + b1A * b7B, (b5A + b5B) * f, (b6A + b6B) * f,
            (b7A + b7B) * f
        ])
        bloopAB = np.array([
            1., 1./2. * (b1A + b1B), 1./2. * (b2A + b2B),
            1./2. * (b3A + b3B), 1./2. * (b4A + b4B),
            b1A * b1B, 1./2. * (b1A * b2B + b1B * b2A),
            1./2. * (b1A * b3B + b1B * b3A),
            1./2. * (b1A * b4B + b1B * b4A),
            b2A * b2B, 1./2. * (b2A * b4B + b2B * b4A), b4A * b4B
        ])
        Ps0 = np.einsum('b,lbx->lx', b11AB, self.P11l)
        Ps1 = (np.einsum('b,lbx->lx', bloopAB, self.Ploopl)
               + np.einsum('b,lbx->lx', bctAB, self.Pctl))
        self.fullPs = Ps0 + Ps1


@dataclass
class EFTTheoryState:
    can_compute: bool = False
    cross: bool = False
    chained: bool = False
    projection: bool = False
    window: bool = False
    fiber: bool = False
    binning: bool = False
    ic: bool = False


class EFTTheory:
    state: EFTTheoryState
    co: pybird.Common
    nonlinear: pybird.NonLinear
    resum: pybird.Resum
    projection: Optional[pybird.Projection]
    bird: Optional[BirdPlus]
    bolzmann_provider: Optional[BoltzmannProvider]
    print_info: Callable[[str], None]
    # for computing
    z: float
    ls: List[int]
    ktrust: Optional[float]
    fs: Optional[float]
    Dfc: Optional[float]

    def __init__(
        self,
        z: float,
        cache_dir_path: Path,
        optiresum: bool = False,
        Nl: int = 2,
        cross: bool = False,
        chained: bool = False,
        projection_config: Optional[ProjectionConfig] = None,
        bolzmann_provider: Optional[BoltzmannProvider] = None,
        print_info=print
    ) -> None:
        self.print_info = print_info
        state = EFTTheoryState()
        self.state = state

        print_info('==========================>')
        print_info(f'Initializing EFTTheory:')
        self.z = z
        print_info(f'effective redshift: {self.z:.3f}')
        if cross:
            state.cross = True
            print_info(f'computing cross power spectrum')
        if chained:
            state.chained = True
            print_info(f'computing chained power spectrum up to {Nl}')
            Nl = Nl + 1
        ls = [2 * i for i in range(Nl)]
        self.ls = ls
        print_info(f'computing power spectrum ls = {ls}')
        self.co = pybird.Common(Nl=Nl, optiresum=optiresum)
        if optiresum:
            print_info('resummation: optimized')
        else:
            print_info('resummation: full')
        self.nonlinear = pybird.NonLinear(
            load=True, save=True, co=self.co,
            path=str(cache_dir_path.resolve())
        )
        self.resum = pybird.Resum(co=self.co)

        if bolzmann_provider is not None:
            state.can_compute = True
        self.bolzmann_provider = bolzmann_provider

        self.projection = None
        self.bird = None
        self.ktrust, self.fs, self.Dfc = None, None, None
        if projection_config is not None:
            state.projection = True
            self.set_projection(**projection_config)
        print_info('<==========================')

    def set_bolzman_provider(self, provider):
        self.bolzmann_provider = provider
        self.state.can_compute = True

    def set_projection(
        self,
        Om_AP: float,
        z_AP: float,
        rdrag_fid: Optional[float] = None,
        kdata: Optional[NDArray] = None,
        windows_fourier_path: Optional[Path] = None,
        windows_configspace_path: Optional[Path] = None,
        ktrust: Optional[float] = None,
        fs: Optional[float] = None,
        Dfc: Optional[float] = None,
        binning: bool = False,
        integral_constraint_path: Optional[Path] = None,
        shotnoise: float = 0.
    ) -> None:
        self.print_info('start setting projection:')
        state = self.state

        if kdata is None:
            self.print_info('using default kdata')
            kdata = np.linspace(0.005, 0.3, 50)

        if windows_fourier_path is None:
            self.print_info('window: off')
            if rdrag_fid is not None:
                self.print_info(f'fiducial rdrag = {rdrag_fid}')
            projection = pybird.Projection(
                kdata, Om_AP, z_AP, co=self.co, rdrag_fid=rdrag_fid
            )
        else:
            self.print_info('window: on')
            state.window = True
            if binning:
                self.print_info('binning: on')
                state.binning = True
            projection = pybird.Projection(
                kdata, Om_AP, z_AP, co=self.co,
                window_fourier_name=windows_fourier_path.name,
                path_to_window=str(windows_fourier_path.resolve().parent),
                window_configspace_file=windows_configspace_path,
                binning=state.binning, rdrag_fid=rdrag_fid
            )
        self.projection = projection

        if all([_ is not None for _ in (ktrust, fs, Dfc)]):
            self.print_info('fiber collision correction: on')
            self.ktrust, self.fs, self.Dfc = ktrust, fs, Dfc
            state.fiber = True
        else:
            self.print_info('fiber collision correction: off')

        if integral_constraint_path is not None:
            state.ic = True
            raise NotImplementedError

    def stochastic_terms(self, ks: NDArray, es: Iterable[float]) -> NDArray:
        """compute stochastic terms

        Parameters
        ----------
        es : Iterable[float]
            c_{e,0}/n_d, c_{mono}/n_d/k_m^2, c_{quad}/n_d/k_m^2
        """
        e1, e2, e3 = es
        mono = e1 + e2 * ks**2
        quad = e3 * ks**2
        if self.co.Nl == 1:
            return mono
        elif self.co.Nl == 2:
            return np.vstack((mono, quad))
        else:
            return np.vstack((mono, quad, np.zeros(mono.shape)))

    def stochastic_terms_with_windows(self, es: Iterable[float]) -> NDArray:
        assert self.projection is not None
        pkl = self.stochastic_terms(self.projection.p, es)
        out = np.einsum('alkp,lp->ak', self.projection.Waldk, pkl)
        out = interp1d(
            self.co.k, out, axis=-1, kind='cubic')(self.projection.kout)
        return out

    def theory_vector(
        self,
        bsA: Iterable[float], *,
        bsB: Optional[Iterable[float]] = None,
        es: Iterable[float] = (0., 0., 0.),
        rdrag_true: Optional[float] = None
    ) -> NDArray:
        if not self.state.can_compute:
            raise ValueError("missing bolzmann provider")
        provider = self.bolzmann_provider
        assert provider is not None
        if (self.bird is not None) and (not provider.cosmo_updated()):
            self.bird.setreducePslb(bsA, bsB)
        else:
            kh = np.logspace(-4, 0, 200)
            pkh = provider.interp_pkh(kh)

            h = provider.get_h0()
            H = provider.get_Hubble(self.z) / (h * 100)
            DA = provider.get_angular_diameter_distance(
                self.z) * (h * 100) / 299792.458
            f = provider.get_fsigma8(self.z) / provider.get_sigma8_z(self.z)
            H, DA, f = float(H), float(DA), float(f)
            bird = BirdPlus(
                kh, pkh, f, DA, H, self.z, which='all', co=self.co)
            self.nonlinear.PsCf(bird)
            bird.setPsCfl()
            self.resum.Ps(bird)
            state = self.state
            if self.projection is not None:
                self.projection.AP(bird, rdrag_true=rdrag_true)
                if state.window:
                    self.projection.Window(bird)
                if state.fiber:
                    self.projection.fibcolWindow(
                        bird, ktrust=self.ktrust, fs=self.fs, Dfc=self.Dfc)
                if state.binning:
                    self.projection.kbinning(bird)
                else:
                    self.projection.kdata(bird)
            bird.setreducePslb(bsA, bsB)
            self.bird = bird

        out: NDArray = self.bird.fullPs.copy()
        if self.projection is not None:
            if self.state.window:
                out += self.stochastic_terms_with_windows(es)
            else:
                out += self.stochastic_terms(self.projection.kout, es)
        else:
            out += self.stochastic_terms(pybird.kbird, es)

        if self.state.chained:
            newout = out.copy()
            for i in range(self.co.Nl - 1):
                newout[i, :] = out[i, :] - chain_coeff(2 * i) * out[i + 1, :]
            out = newout[:-1, :]
        return out.reshape(-1)
