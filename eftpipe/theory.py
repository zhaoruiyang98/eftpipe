# global
import numpy as np
import sys
from cobaya.theory import Provider
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
    Dict,
    cast,
)
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
# local
from eftpipe.pybird import pybird
from eftpipe.interface import CobayaCambProvider
from eftpipe.typing import (
    ProjectionConfig,
    BoltzmannProvider,
    Location,
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
        bsB: Optional[Iterable[float]] = None,
        es: Iterable[float] = (0., 0., 0.)
    ) -> None:
        """apply counter terms and bind fullPs to self

        Parameters
        ----------
        bsA : Iterable[float]
            b_1, b_2, b_3, b_4, c_{ct}, c_{r,1}, c{r,2}
        bsB : Optional[Iterable[float]], optional
            the same as bsA, but for tracer A, by default None
            leave it default will compute auto power spectrum
        es : Iterable[float], optional
            c_{e,0}, c_{mono}, c_{quad}, by default zeros
        """
        kmA, ndA, kmB, ndB = self.co.kmA, self.co.ndA, self.co.kmB, self.co.ndB
        b1A, b2A, b3A, b4A, cctA, cr1A, cr2A = bsA
        if bsB is None:
            bsB = bsA
        b1B, b2B, b3B, b4B, cctB, cr1B, cr2B = bsB
        f = self.f
        ce0, cemono, cequad = es

        # cct -> cct / km**2, cr1 -> cr1 / km**2, cr2 -> cr2 / km**2
        # ce0 -> ce0 / nd, cemono -> cemono / nd / km**2, cequad -> cequad / nd / km**2
        b11AB = np.array([b1A * b1B, (b1A + b1B) * f, f**2])
        bctAB = np.array([
            b1A * cctB / kmB**2 + b1B * cctA / kmA**2,
            b1B * cr1A / kmA**2 + b1A * cr1B / kmB**2,
            b1B * cr2A / kmA**2 + b1A * cr2B / kmB**2,
            (cctA / kmA**2 + cctB / kmB**2) * f,
            (cr1A / kmA**2 + cr1B / kmB**2) * f,
            (cr2A / kmA**2 + cr2B / kmB**2) * f,
        ])
        bloopAB = np.array([
            1.,
            1. / 2. * (b1A + b1B),
            1. / 2. * (b2A + b2B),
            1. / 2. * (b3A + b3B),
            1. / 2. * (b4A + b4B),
            b1A * b1B,
            1. / 2. * (b1A * b2B + b1B * b2A),
            1. / 2. * (b1A * b3B + b1B * b3A),
            1. / 2. * (b1A * b4B + b1B * b4A),
            b2A * b2B,
            1. / 2. * (b2A * b4B + b2B * b4A),
            b4A * b4B
        ])
        xfactor1 = 0.5 * (1.0 / ndA + 1.0 / ndB)
        xfactor2 = 0.5 * (1.0 / ndA / kmA**2 + 1.0 / ndB / kmB**2)
        bstAB = np.array(
            [ce0 * xfactor1, cemono * xfactor2, cequad * xfactor2])
        Ps0 = np.einsum('b,lbx->lx', b11AB, self.P11l)
        Ps1 = (np.einsum('b,lbx->lx', bloopAB, self.Ploopl)
               + np.einsum('b,lbx->lx', bctAB, self.Pctl))
        Ps2 = np.einsum('b,lbx->lx', bstAB, self.Pstl)
        self.fullPs = Ps0 + Ps1 + Ps2


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
        cache_dir_path: Location,
        kmA: float,
        ndA: float,
        kmB: Optional[float] = None,
        ndB: Optional[float] = None,
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
        self.co = pybird.Common(
            Nl=Nl, optiresum=optiresum, kmA=kmA, ndA=ndA, kmB=kmB, ndB=ndB)
        if optiresum:
            print_info('resummation: optimized')
        else:
            print_info('resummation: full')
        self.nonlinear = pybird.NonLinear(
            load=True, save=True, co=self.co,
            path=str(Path(cache_dir_path).resolve())
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
        windows_fourier_path: Optional[Location] = None,
        windows_configspace_path: Optional[Location] = None,
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

        if rdrag_fid is not None:
            self.print_info(f'fiducial rdrag = {rdrag_fid}')

        if binning:
            self.print_info('binning: on')
            state.binning = True

        if windows_fourier_path is None:
            self.print_info('window: off')
            projection = pybird.Projection(
                kdata, Om_AP, z_AP, co=self.co,
                binning=state.binning, rdrag_fid=rdrag_fid
            )
        else:
            self.print_info('window: on')
            state.window = True
            projection = pybird.Projection(
                kdata, Om_AP, z_AP, co=self.co,
                window_fourier_name=Path(windows_fourier_path).name,
                path_to_window=str(
                    Path(windows_fourier_path).resolve().parent
                ),
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

    def theory_vector(
        self,
        bsA: Iterable[float], *,
        bsB: Optional[Iterable[float]] = None,
        es: Iterable[float] = (0., 0., 0.),
    ) -> NDArray:
        if not self.state.can_compute:
            raise ValueError("missing bolzmann provider")
        provider = self.bolzmann_provider
        assert provider is not None
        if (not provider.cosmo_updated()) and (self.bird is not None):
            self.bird.setreducePslb(bsA, bsB, es=es)
        else:
            kh = np.logspace(-4, 0, 200)
            pkh = provider.interp_pkh(kh)

            H, DA, f = (
                provider.get_H(self.z),
                provider.get_DA(self.z),
                provider.get_f(self.z)
            )
            bird = BirdPlus(
                kh, pkh, f, DA, H, self.z, which='all', co=self.co)
            self.nonlinear.PsCf(bird)
            bird.setPsCfl()
            self.resum.Ps(bird)
            state = self.state
            if self.projection is not None:
                self.projection.AP(bird, rdrag_true=provider.get_rdrag())
                if state.window:
                    self.projection.Window(bird)
                if state.fiber:
                    self.projection.fibcolWindow(
                        bird, ktrust=self.ktrust, fs=self.fs, Dfc=self.Dfc)
                if state.binning:
                    self.projection.kbinning(bird)
                else:
                    self.projection.kdata(bird)
            bird.setreducePslb(bsA, bsB, es=es)
            self.bird = bird

        out: NDArray = self.bird.fullPs.copy()

        if self.state.chained:
            newout = out.copy()
            for i in range(self.co.Nl - 1):
                newout[i, :] = out[i, :] - chain_coeff(2 * i) * out[i + 1, :]
            out = newout[:-1, :]
        return out.reshape(-1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~vector theory~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class SingleTracerEFT:
    theory: EFTTheory
    prefix: str

    def __init__(self, theory: EFTTheory, prefix: str = "") -> None:
        self.theory = theory
        self.prefix = prefix
        self._set_required_params()

    def set_provider(self, provider: Provider) -> None:
        self.theory.set_bolzman_provider(
            CobayaCambProvider(provider, self.theory.z)
        )

    def required_params(self) -> Dict[str, Any]:
        return self._required_params

    def _set_required_params(self) -> None:
        z = self.theory.z
        extra_zs = [] if z == 0. else [0.]
        requires = {
            'Pk_grid': {
                'nonlinear': False,
                'z': [z],
                'k_max': 100
            },
            'Hubble': {'z': extra_zs + [z]},
            'angular_diameter_distance': {'z': [z]},
            'fsigma8': {'z': [z]},
            'sigma8_z': {'z': [z]},
            'rdrag': None
        }
        eft_params = [
            self.prefix + name for name in
            ('b1', 'b2', 'b3', 'b4',
             'cct', 'cr1', 'cr2',
             'ce0', 'cemono', 'cequad')
        ]
        eft_requires = dict(
            zip(eft_params, [None for _ in range(len(eft_params))])
        )
        requires.update(eft_requires)
        self._required_params = requires

    def theory_vector(self, all_params_dict: Dict[str, Any]) -> NDArray:
        prefix = self.prefix
        (
            b1, b2, b3, b4,
            cct, cr1, cr2,
            ce0, cemono, cequad,
        ) = [all_params_dict[prefix + name] for name in (
            'b1', 'b2', 'b3', 'b4',
            'cct', 'cr1', 'cr2',
            'ce0', 'cemono', 'cequad',
        )]
        bs = [b1, b2, b3, b4, cct, cr1, cr2]
        es = [ce0, cemono, cequad]
        return self.theory.theory_vector(bs, es=es)


class TwoTracerEFT:
    theories: List[EFTTheory]
    prefixes: List[str]

    def __init__(self, theories: List[EFTTheory], prefixes: List[str]) -> None:
        self.theories = theories
        if len(set(prefixes)) != 2:
            raise ValueError('TwoTracerEFT needs two different prefixes')
        self.prefixes = prefixes
        self._set_required_params()

    def set_provider(self, provider: Provider) -> None:
        for theory in self.theories:
            theory.set_bolzman_provider(
                CobayaCambProvider(provider, theory.z)
            )

    def required_params(self) -> Dict[str, Any]:
        return self._required_params

    def _set_required_params(self) -> None:
        zs = [theory.z for theory in self.theories]
        zs = list(set(zs))
        extra_zs = [] if 0. in zs else [0.]
        requires = {
            'Pk_grid': {
                'nonlinear': False,
                'z': zs,
                'k_max': 100
            },
            'Hubble': {'z': extra_zs + zs},
            'angular_diameter_distance': {'z': zs},
            'fsigma8': {'z': zs},
            'sigma8_z': {'z': zs},
            'rdrag': None
        }
        eft_params_names = [
            'b1', 'b2', 'b3', 'b4',
            'cct', 'cr1', 'cr2',
            'ce0', 'cemono', 'cequad',
        ]
        eft_params = []
        for prefix in self.prefixes:
            eft_params += [prefix + name for name in eft_params_names]
        eft_requires = dict(
            zip(eft_params, [None for _ in range(len(eft_params))])
        )
        requires.update(eft_requires)
        self._required_params = requires

    def theory_vector(self, all_params_dict: Dict[str, Any]) -> NDArray:
        vectors = []
        for (prefix, theory) in zip(self.prefixes, self.theories):
            (
                b1, b2, b3, b4,
                cct, cr1, cr2,
                ce0, cemono, cequad,
            ) = [all_params_dict[prefix + name] for name in (
                'b1', 'b2', 'b3', 'b4',
                'cct', 'cr1', 'cr2',
                'ce0', 'cemono', 'cequad',
            )]
            bs = [b1, b2, b3, b4, cct, cr1, cr2]
            es = [ce0, cemono, cequad]
            vectors.append(theory.theory_vector(bs, es=es))
        return np.hstack(vectors)


class TwoTracerCrossEFT:
    _index_mapping: Dict[str, int]
    theories: List[EFTTheory]
    prefixes: List[str]

    def __init__(self, theories: List[EFTTheory], prefixes: List[str]) -> None:
        if len(theories) != 3:
            raise ValueError('TwoTracerCrossEFT needs three EFTTheory objects')
        ncross = 0
        for theory in theories:
            if theory.state.cross:
                ncross += 1
        if ncross != 1:
            raise ValueError(
                'TwoTracerCrossEFT needs exactly one cross EFTTheory object')
        self.theories = theories
        if len(set(prefixes)) != 3:
            raise ValueError(
                'TwoTracerCrossEFT needs three different prefixes')
        self.prefixes = prefixes
        self._set_required_params()
        self._set_index_mapping()

    def set_provider(self, provider: Provider) -> None:
        for theory in self.theories:
            theory.set_bolzman_provider(
                CobayaCambProvider(provider, theory.z)
            )

    def required_params(self) -> Dict[str, Any]:
        return self._required_params

    def _set_index_mapping(self) -> None:
        index_mapping = cast(
            Dict[Literal['A', 'B', 'x'], int],
            {'A': None, 'B': None, 'x': None})
        for i, theory in enumerate(self.theories):
            if not theory.state.cross:
                if index_mapping['A'] is None:
                    index_mapping['A'] = i
                else:
                    index_mapping['B'] = i
            else:
                index_mapping['x'] = i
        self._index_mapping = index_mapping

    def _set_required_params(self) -> None:
        zs = [theory.z for theory in self.theories]
        zs = list(set(zs))
        extra_zs = [] if 0. in zs else [0.]
        requires = {
            'Pk_grid': {
                'nonlinear': False,
                'z': zs,
                'k_max': 100
            },
            'Hubble': {'z': extra_zs + zs},
            'angular_diameter_distance': {'z': zs},
            'fsigma8': {'z': zs},
            'sigma8_z': {'z': zs},
            'rdrag': None
        }
        eft_params_names = [
            'b1', 'b2', 'b3', 'b4',
            'cct', 'cr1', 'cr2',
            'ce0', 'cemono', 'cequad',
        ]
        cross_params_names = ['ce0', 'cemono', 'cequad']
        eft_params = []
        for prefix, theory in zip(self.prefixes, self.theories):
            params_list = eft_params_names
            if theory.state.cross:
                params_list = cross_params_names
            eft_params += [prefix + name for name in params_list]
        eft_requires = dict(
            zip(eft_params, [None for _ in range(len(eft_params))])
        )
        requires.update(eft_requires)
        self._required_params = requires

    def theory_vector(self, all_params_dict: Dict[str, Any]) -> NDArray:
        # TODO: stupid implementation, should be improved
        eft_params_names = [
            'b1', 'b2', 'b3', 'b4',
            'cct', 'cr1', 'cr2',
            'ce0', 'cemono', 'cequad',
        ]
        cross_params_names = ['ce0', 'cemono', 'cequad']
        Aindex, Bindex, xindex = [
            self._index_mapping[key] for key in ('A', 'B', 'x')
        ]
        prefixA = self.prefixes[Aindex]
        prefixB = self.prefixes[Bindex]
        prefixx = self.prefixes[xindex]
        (
            b1A, b2A, b3A, b4A,
            cctA, cr1A, cr2A,
            ce0A, cemonoA, cequadA,
        ) = [all_params_dict[prefixA + name] for name in eft_params_names]
        bsA = [b1A, b2A, b3A, b4A, cctA, cr1A, cr2A]
        esA = [ce0A, cemonoA, cequadA]
        (
            b1B, b2B, b3B, b4B,
            cctB, cr1B, cr2B,
            ce0B, cemonoB, cequadB,
        ) = [all_params_dict[prefixB + name] for name in eft_params_names]
        bsB = [b1B, b2B, b3B, b4B, cctB, cr1B, cr2B]
        esB = [ce0B, cemonoB, cequadB]
        ce0x, cemonox, cequadx = [
            all_params_dict[prefixx + name] for name in cross_params_names]
        esx = [ce0x, cemonox, cequadx]

        theory_vectorA = self.theories[Aindex].theory_vector(bsA, es=esA)
        theory_vectorB = self.theories[Bindex].theory_vector(bsB, es=esB)
        theory_vectorx = self.theories[xindex].theory_vector(
            bsA, bsB=bsB, es=esx)
        vectors = cast(List[NDArray], [None, None, None])
        vectors[Aindex] = theory_vectorA
        vectors[Bindex] = theory_vectorB
        vectors[xindex] = theory_vectorx
        return np.hstack(vectors)
