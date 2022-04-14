# global
import inspect
import sys
import numpy as np
from pathlib import Path
from typing import (
    Any,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
)
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
from numpy import ndarray as NDArray
from scipy.special import legendre
from cobaya.log import HasLogger
from cobaya.theory import Provider
# local
from eftpipe.interface import CambProvider
from eftpipe.interface import CobayaCambProvider
from eftpipe.interface import CobayaClassyProvider
from eftpipe.pybird import pybird
from eftpipe.typing import (
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
        es: Iterable[float] = (0., 0., 0.),
        marg: bool = False,
        chained: bool = False,
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

        if marg:
            self.setPG(b1A, b1B, chained=chained)

    def setPG(self, b1A, b1B, chained=False):
        f = self.f
        kmA, ndA, kmB, ndB = self.co.kmA, self.co.ndA, self.co.kmB, self.co.ndB
        # b3A, cctA, cr1A, cr2A, b3B, cctB, cr1B, cr2B, ce0, cemono, cequad
        Nmarg = 4 * 2 + 3
        Nl = self.P11l.shape[0]
        Nk = self.P11l.shape[-1]
        PG = np.empty((Nmarg, Nl, Nk), dtype=np.float64)
        # b3A
        PG[0, :, :] = (
            1 / 2 * self.Ploopl[:, 3, :]
            + 1 / 2 * b1B * self.Ploopl[:, 7, :])
        # cctA
        PG[1, :, :] = (
            b1B / kmA**2 * self.Pctl[:, 0, :]
            + f / kmA**2 * self.Pctl[:, 3, :])
        # cr1A
        PG[2, :, :] = (
            b1B / kmA**2 * self.Pctl[:, 1, :]
            + f / kmA**2 * self.Pctl[:, 4, :])
        # cr2A
        PG[3, :, :] = (
            b1B / kmA**2 * self.Pctl[:, 2, :]
            + f / kmA**2 * self.Pctl[:, 5, :])
        # b3B
        PG[4, :, :] = (
            1 / 2 * self.Ploopl[:, 3, :]
            + 1 / 2 * b1A * self.Ploopl[:, 7, :])
        # cctB
        PG[5, :, :] = (
            b1A / kmB**2 * self.Pctl[:, 0, :]
            + f / kmB**2 * self.Pctl[:, 3, :])
        # cr1B
        PG[6, :, :] = (
            b1A / kmB**2 * self.Pctl[:, 1, :]
            + f / kmB**2 * self.Pctl[:, 4, :])
        # cr2B
        PG[7, :, :] = (
            b1A / kmB**2 * self.Pctl[:, 2, :]
            + f / kmB**2 * self.Pctl[:, 5, :])
        xfactor1 = 0.5 * (1.0 / ndA + 1.0 / ndB)
        xfactor2 = 0.5 * (1.0 / ndA / kmA**2 + 1.0 / ndB / kmB**2)
        # ce0
        PG[8, :, :] = self.Pstl[:, 0, :] * xfactor1
        # cemono
        PG[9, :, :] = self.Pstl[:, 1, :] * xfactor2
        # cequad
        PG[10, :, :] = self.Pstl[:, 2, :] * xfactor2
        self.PG = PG
        if chained:
            PG = self.PG
            newPG = np.empty_like(PG)
            for i in range(self.co.Nl - 1):
                newPG[:, i, :] = \
                    PG[:, i, :] - chain_coeff(2 * i) * PG[:, i + 1, :]
            self.PG = newPG[:, :-1, :]


class ConfigTwice(Exception):
    pass


class InvalidConfig(Exception):
    pass


class Operator:
    def __init__(self, setfunc, apply, **override) -> None:
        if not inspect.isclass(setfunc):
            raise NotImplementedError
        self.setfunc = setfunc
        if apply.__name__ not in dir(setfunc):
            raise NotImplementedError
        self.apply_name = apply.__name__

        args: List[str] = []
        kwargs: Dict[str, Any] = {}
        for k, v in inspect.signature(setfunc).parameters.items():
            if v.kind is v.VAR_KEYWORD:
                continue
            if v.kind is v.VAR_POSITIONAL:
                raise NotImplementedError(
                    f"{self.__class__.__name__} does not support *args")
            default = v.default
            if default is v.empty:
                args.append(k)
            else:
                kwargs[k] = default
        for k, v in override.items():
            if k in args:
                args.remove(k)
            kwargs[k] = v
        self.args = args
        self.kwargs = kwargs
        self.obj = None

    def config(self, dct: Dict[str, Any]) -> None:
        dct = dct if dct else {}
        if self.obj is not None:
            raise ConfigTwice(
                f"{self.__class__.__name__} cannot be configured twice")
        self.kwargs.update(dct)
        for k in self.args:
            if k not in self.kwargs.keys():
                raise InvalidConfig(f"missing positional argument {k}")

        self.obj = self.setfunc(**self.kwargs)

    def apply(self, x):
        return getattr(self.obj, self.apply_name)(x)


class IncompleteTheory(Exception):
    pass


class EFTTheory(HasLogger):
    def __init__(
        self,
        z: float,
        cache_dir_path: Optional[Location] = None,
        km: Optional[float] = None,
        nd: Optional[float] = None,
        kmA: Optional[float] = None,
        ndA: Optional[float] = None,
        kmB: Optional[float] = None,
        ndB: Optional[float] = None,
        cross: bool = False,
        Nl: int = 2,
        optiresum: bool = False,
        chained: bool = False,
        with_IRresum: bool = True,
        with_APeffect: bool = False,
        with_window: bool = False,
        with_fiber: bool = False,
        with_binning: bool = True,
        boltzmann_provider: Optional[BoltzmannProvider] = None,
        config_settings: Dict[str, Any] = None, # type: ignore
    ) -> None:
        self.set_logger(name="eftpipe.EFTTheory")

        self.z = z
        self.mpi_info("effective redshift: %.3f", self.z)

        if (km is None or nd is None) and (kmA is None or ndA is None):
            raise ValueError("expect parameters pair (km, nd) or (kmA, ndA)")
        if km and nd and kmA and ndA:
            raise ValueError(
                "cannot pass both parameters pair (km, nd) and (kmdA, ndA)")
        kmA = kmA if kmA else km
        ndA = ndA if ndA else nd
        if cross and (kmB is None or ndB is None):
            raise ValueError("missing parameters pair (kmB, ndB)")
        self.cross = cross
        if cross:
            self.mpi_info("compute cross power spectrum")
        self.chained = chained
        if chained:
            self.mpi_info("compute chained power spectrum up to %d", Nl)
            Nl = Nl + 1
        ls = [2 * i for i in range(Nl)]
        self.mpi_info("compute power spectrum multipoles ls = %s", ls)
        self.co = pybird.Common(
            Nl=Nl, optiresum=optiresum, kmA=kmA, ndA=ndA, kmB=kmB, ndB=ndB)
        self.bird: Optional[BirdPlus] = None

        if cache_dir_path is None:
            cache_dir_path = Path.cwd() / 'cache'
        self.nonlinear = pybird.NonLinear(
            load=True, save=True, co=self.co,
            path=str(Path(cache_dir_path).resolve())
        )

        self.boltzmann_provider = boltzmann_provider
        self.operators: Dict[str, Operator] = {}
        self.configured: bool = False
        self.marg = False # default False but may be modified by other classes
        # order is important
        if with_IRresum:
            self.operators["IRresum"] = Operator(
                pybird.Resum, pybird.Resum.Ps, co=self.co)
            self.mpi_info(
                "IRresum enabled: %s", "optimized" if optiresum else "full")
        if with_APeffect:
            self.operators["APeffect"] = Operator(
                pybird.APeffect, pybird.APeffect.AP, co=self.co)
        if with_window:
            self.operators["window"] = Operator(
                pybird.Window, pybird.Window.Window, co=self.co)
            self.mpi_info("APeffect enabled")
        if with_fiber:
            self.operators["fiber"] = Operator(
                pybird.FiberCollision, pybird.FiberCollision.fibcolWindow,
                co=self.co
            )
            self.mpi_info("fiber enabled")
        if with_binning:
            self.operators["binning"] = Operator(
                pybird.Binning, pybird.Binning.match, co=self.co)
            self.mpi_info("binning enabled")
        if config_settings:
            self.set_config(config_settings)

    def set_boltzmann_provider(self, provider: BoltzmannProvider):
        self.boltzmann_provider = provider

    def set_config(self, dct):
        diff = set(dct.keys()).difference(self.operators.keys())
        if diff:
            self.mpi_warning(
                "configuration contains %s, "
                "but the corresponding operators are not loaded or not exist",
                diff,
            )
        for k, op in self.operators.items():
            op.config(dct.get(k, {}))
        self.configured = True

    def theory_vector(
        self,
        bsA: Iterable[float], *,
        bsB: Optional[Iterable[float]] = None,
        es: Iterable[float] = (0., 0., 0.),
    ) -> NDArray:
        if self.boltzmann_provider is None:
            raise IncompleteTheory("missing boltzmann_provider")
        if not self.configured:
            raise IncompleteTheory("theory has not been configured")
        provider = self.boltzmann_provider

        # cosmo_updated should be called first to record the cosmology
        if provider.cosmo_updated() or self.bird is None:
            # TODO: test larger kh range
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
            for op in self.operators.values():
                op.apply(bird)
            bird.setreducePslb(
                bsA, bsB, es=es, marg=self.marg, chained=self.chained)
            self.bird = bird
        else:
            self.bird.setreducePslb(
                bsA, bsB, es=es, marg=self.marg, chained=self.chained)

        out: NDArray = self.bird.fullPs.copy()

        if self.chained:
            newout = np.empty_like(out)
            for i in range(self.co.Nl - 1):
                newout[i, :] = out[i, :] - chain_coeff(2 * i) * out[i + 1, :]
            out = newout[:-1, :]
        return out.reshape(-1)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~vector theory~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class SingleTracerEFT(HasLogger):
    """A wrapped theory class to interact with cobaya

    Parameters
    ----------
    theory: EFTTheory
        basic theory
    prefix: str
        prefix of EFT parameters, by default ""
    provider: "camb" or "classy"
        use cobaya's camb or cobaya's classy, by default "camb"
    """

    def __init__(
        self,
        theory: EFTTheory, prefix: str = "",
        provider: Literal["camb", "classy"] = "camb",
    ) -> None:
        self.set_logger(name="eftpipe.SingleTracerEFT")

        self.theory = theory
        self.prefix = prefix
        self.can_marg = True
        self.provider = provider
        if provider not in ("camb", "classy"):
            raise ValueError("only support provider: camb or provider: classy")
        self.mpi_info("using provider %s", self.provider)

    def set_provider(self, provider: Provider) -> None:
        if self.provider == "camb":
            self.theory.set_boltzmann_provider(
                CobayaCambProvider(provider, self.theory.z))
        elif self.provider == "classy":
            self.theory.set_boltzmann_provider(
                CobayaClassyProvider(provider, self.theory.z))

    def set_camb_provider(self, **kwargs) -> None:
        self.theory.set_boltzmann_provider(
            CambProvider(z=self.theory.z, **kwargs))

    def required_params(self) -> Dict[str, Any]:
        z = self.theory.z
        extra_zs = [] if z == 0. else [0.]
        if self.provider == "camb":
            requires = {
                'Pk_grid': {
                    'nonlinear': False,
                    'z': [z],
                    'k_max': 5,
                },
                'Hubble': {'z': extra_zs + [z]},
                'angular_diameter_distance': {'z': [z]},
                'fsigma8': {'z': [z]},
                'sigma8_z': {'z': [z]},
                'rdrag': None,
            }
        else:
            requires = {
                'Pk_interpolator': {
                    'nonlinear': False,
                    'z': [z],
                    'k_max': 5,
                },
                'Hubble': {'z': extra_zs + [z]},
                'angular_diameter_distance': {'z': [z]},
                'rdrag': None,
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
        return requires

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

    def can_marg_params(self) -> List[str]:
        return [
            self.prefix + x
            for x in ('b3', 'cct', 'cr1', 'cr2', 'ce0', 'cemono', 'cequad')
        ]

    def set_marg(self, prior: Dict[str, Any]) -> None:
        self.theory.marg = True
        all_params = self.can_marg_params()
        marginds = [all_params.index(name) for name in prior.keys()]
        margcoef = np.ones(len(marginds), dtype=np.float64)
        for i, index in enumerate(marginds):
            if index < 4:
                margcoef[i] = 2
            else:
                marginds[i] += 4
        self.marginds = marginds
        self.margcoef = margcoef

    def PG(self, all_params_dict: Dict[str, Any]) -> NDArray:
        out = np.vstack(
            [self.theory.bird.PG[i, ...].reshape(-1) for i in self.marginds])  # type: ignore
        return out * self.margcoef[:, None]

    def PNG(self, all_params_dict: Dict[str, Any]) -> NDArray:
        return self.theory_vector(all_params_dict)


class TwoTracerEFT(HasLogger):
    """A wrapped theory class to interact with cobaya

    Parameters
    ----------
    theories: list[EFTTheory]
        basic theories list
    prefixes: list[str]
        EFT parameters' prefixes list
    provider: "camb" or "classy"
        use cobaya's camb or cobaya's classy, default is "camb"
    """

    def __init__(
        self,
        theories: List[EFTTheory], prefixes: List[str],
        provider: Literal["camb", "classy"] = "camb",
    ) -> None:
        self.set_logger(name="eftpipe.TwoTracerEFT")

        self.theories = theories
        if len(set(prefixes)) != 2:
            raise ValueError('TwoTracerEFT needs two different prefixes')
        self.prefixes = prefixes
        self.can_marg = True
        self.provider = provider
        if provider not in ("camb", "classy"):
            raise ValueError("only support provider: camb or provider: classy")
        self.mpi_info("using provider %s", self.provider)

    def set_provider(self, provider: Provider) -> None:
        for theory in self.theories:
            if self.provider == "camb":
                theory.set_boltzmann_provider(
                    CobayaCambProvider(provider, theory.z))
            elif self.provider == "classy":
                theory.set_boltzmann_provider(
                    CobayaClassyProvider(provider, theory.z))

    def set_camb_provider(self, **kwargs) -> None:
        for theory in self.theories:
            theory.set_boltzmann_provider(
                CambProvider(z=theory.z, **kwargs))

    def required_params(self) -> Dict[str, Any]:
        zs = [theory.z for theory in self.theories]
        zs = list(set(zs))
        extra_zs = [] if 0. in zs else [0.]
        if self.provider == "camb":
            requires = {
                'Pk_grid': {
                    'nonlinear': False,
                    'z': zs,
                    'k_max': 5
                },
                'Hubble': {'z': extra_zs + zs},
                'angular_diameter_distance': {'z': zs},
                'fsigma8': {'z': zs},
                'sigma8_z': {'z': zs},
                'rdrag': None
            }
        else:
            requires = {
                "Pk_interpolator": {
                    "nonlinear": False,
                    "z": zs,
                    "k_max": 5,
                },
                "Hubble": {"z": extra_zs + zs},
                "angular_diameter_distance": {"z": zs},
                "rdrag": None,
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
        return requires

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

    def can_marg_params(self) -> List[str]:
        return [
            prefix + x
            for prefix in self.prefixes
            for x in ('b3', 'cct', 'cr1', 'cr2', 'ce0', 'cemono', 'cequad')
        ]

    def set_marg(self, prior: Dict[str, Any]) -> None:
        all_paramsA = [
            self.prefixes[0] + x
            for x in ('b3', 'cct', 'cr1', 'cr2', 'ce0', 'cemono', 'cequad')
        ]
        all_paramsB = [
            self.prefixes[1] + x
            for x in ('b3', 'cct', 'cr1', 'cr2', 'ce0', 'cemono', 'cequad')
        ]
        margindsA = [
            all_paramsA.index(name)
            for name in prior.keys()
            if name in all_paramsA]
        margindsB = [
            all_paramsB.index(name)
            for name in prior.keys()
            if name in all_paramsB]
        margcoefA = np.ones(len(margindsA), dtype=np.float64)
        margcoefB = np.ones(len(margindsB), dtype=np.float64)
        for theory, inds in zip(self.theories, (margindsA, margindsB)):
            if len(inds) != 0:
                theory.marg = True
        for marginds, margcoef in zip(
                (margindsA, margindsB), (margcoefA, margcoefB)):
            for i, index in enumerate(marginds):
                if index < 4:
                    margcoef[i] = 2
                else:
                    marginds[i] += 4
        self.margindsA = margindsA
        self.marfcoefA = margcoefA
        self.margindsB = margindsB
        self.margcoefB = margcoefB

    def PG(self, all_params_dict: Dict[str, Any]) -> NDArray:
        nvec = [np.prod(theory.bird.PG.shape[1:])  # type: ignore
                for theory in self.theories]
        pad_widths = [
            ((0, 0), (0, nvec[1])),
            ((0, 0), (nvec[0], 0)),
        ]
        if not self.margindsA:
            pad_widths = [pad_widths[1]]
        if not self.margindsB:
            pad_widths = [pad_widths[0]]
        out = [
            np.vstack([theory.bird.PG[i, ...].reshape(-1)  # type: ignore
                      for i in inds])
            for theory, inds in zip(
                self.theories, (self.margindsA, self.margindsB))
            if inds
        ]
        out = np.vstack([
            np.pad(x, pad) for x, pad in zip(out, pad_widths)
        ])
        return out * np.hstack([self.marfcoefA, self.margcoefB])[:, None]

    def PNG(self, all_params_dict: Dict[str, Any]) -> NDArray:
        return self.theory_vector(all_params_dict)


class TwoTracerCrossEFT(HasLogger):
    """A wrapped theory class to interact with cobaya

    Parameters
    ----------
    theories: list[EFTTheory]
        basic theories list
    prefixes: list[str]
        EFT parameters' prefixes list
    provider: "camb" or "classy"
        use cobaya's camb or cobaya's classy, default is "camb"
    """

    def __init__(
        self,
        theories: List[EFTTheory], prefixes: List[str],
        provider: Literal["camb", "classy"] = "camb",
    ) -> None:
        self.set_logger(name="eftpipe.TwoTracerCrossEFT")

        if len(theories) != 3:
            raise ValueError('TwoTracerCrossEFT needs three EFTTheory objects')
        ncross = 0
        for theory in theories:
            if theory.cross:
                ncross += 1
        if ncross != 1:
            raise ValueError(
                'TwoTracerCrossEFT needs exactly one cross EFTTheory object')
        self.theories = theories
        if len(set(prefixes)) != 3:
            raise ValueError(
                'TwoTracerCrossEFT needs three different prefixes')
        self.prefixes = prefixes
        self._type_to_index: Dict[str, int]
        self._index_to_type: Dict[int, str]
        self._set_index_mapping()
        self.can_marg = True
        self.provider = provider
        if provider not in ("camb", "classy"):
            raise ValueError("only support provider: camb or provider: classy")
        self.mpi_info("using provider %s", self.provider)

    def set_provider(self, provider: Provider) -> None:
        for theory in self.theories:
            if self.provider == "camb":
                theory.set_boltzmann_provider(
                    CobayaCambProvider(provider, theory.z))
            elif self.provider == "classy":
                theory.set_boltzmann_provider(
                    CobayaClassyProvider(provider, theory.z))

    def set_camb_provider(self, **kwargs) -> None:
        for theory in self.theories:
            theory.set_boltzmann_provider(
                CambProvider(z=theory.z, **kwargs))

    def _set_index_mapping(self) -> None:
        type_to_index = cast(
            Dict[Literal['A', 'B', 'x'], int],
            {'A': None, 'B': None, 'x': None})
        for i, theory in enumerate(self.theories):
            if not theory.cross:
                if type_to_index['A'] is None:
                    type_to_index['A'] = i
                else:
                    type_to_index['B'] = i
            else:
                type_to_index['x'] = i
        self._type_to_index = type_to_index

        index_to_type = {}
        for name, index in type_to_index.items():
            index_to_type[index] = name
        self._index_to_type = index_to_type

    def required_params(self) -> Dict[str, Any]:
        zs = [theory.z for theory in self.theories]
        zs = list(set(zs))
        extra_zs = [] if 0. in zs else [0.]
        if self.provider == "camb":
            requires = {
                'Pk_grid': {
                    'nonlinear': False,
                    'z': zs,
                    'k_max': 5,
                },
                'Hubble': {'z': extra_zs + zs},
                'angular_diameter_distance': {'z': zs},
                'fsigma8': {'z': zs},
                'sigma8_z': {'z': zs},
                'rdrag': None,
            }
        else:
            requires = {
                "Pk_interpolator": {
                    "nonlinear": False,
                    "z": zs,
                    "k_max": 5
                },
                "Hubble": {'z': extra_zs + zs},
                "angular_diameter_distance": {'z': zs},
                "rdrag": None,
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
            if theory.cross:
                params_list = cross_params_names
            eft_params += [prefix + name for name in params_list]
        eft_requires = dict(
            zip(eft_params, [None for _ in range(len(eft_params))])
        )
        requires.update(eft_requires)
        return requires

    def theory_vector(self, all_params_dict: Dict[str, Any]) -> NDArray:
        # TODO: stupid implementation, should be improved
        eft_params_names = [
            'b1', 'b2', 'b3', 'b4',
            'cct', 'cr1', 'cr2',
            'ce0', 'cemono', 'cequad',
        ]
        cross_params_names = ['ce0', 'cemono', 'cequad']
        Aindex, Bindex, xindex = [
            self._type_to_index[key] for key in ('A', 'B', 'x')
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

    def can_marg_params(self) -> List[str]:
        out = []
        for i, prefix in enumerate(self.prefixes):
            if self._index_to_type[i] == 'x':
                margnames = ('ce0', 'cemono', 'cequad')
            else:
                margnames = (
                    'b3', 'cct', 'cr1', 'cr2', 'ce0', 'cemono', 'cequad')
            out += [prefix + name for name in margnames]
        return out

    def set_marg(self, prior: Dict[str, Any]) -> None:
        marginds_list, margcoef_list = [], []
        margnames = ('b3', 'cct', 'cr1', 'cr2', 'ce0', 'cemono', 'cequad')
        margnames_nost = ('b3', 'cct', 'cr1', 'cr2')
        xmargnames = ('ce0', 'cemono', 'cequad')
        prefixes = self.prefixes
        indexA, indexB, indexx = [
            self._type_to_index[t] for t in ('A', 'B', 'x')]
        for i, prefix in enumerate(prefixes):
            if self._index_to_type[i] == 'x':
                all_params = [
                    prefixes[indexA] + name for name in margnames_nost]
                all_params += [
                    prefixes[indexB] + name for name in margnames_nost]
                all_params += [prefixes[indexx] + name for name in xmargnames]
            else:
                all_params = [prefix + name for name in margnames]
            marginds = [
                all_params.index(name)
                for name in prior.keys()
                if name in all_params
            ]
            margcoef = np.ones(len(marginds), dtype=np.float64)
            if len(marginds) != 0:
                self.theories[i].marg = True
            if self._index_to_type[i] != 'x':
                for i, index in enumerate(marginds):
                    if index < 4:
                        margcoef[i] = 2
                    else:
                        marginds[i] += 4
            marginds_list.append(marginds)
            margcoef_list.append(margcoef)
        self.marginds_list = marginds_list
        self.margcoef_list = margcoef_list
        bias_marginds_A = [ind for ind in marginds_list[indexA] if ind < 4]
        bias_marginds_A = np.argsort(bias_marginds_A)
        bias_marginds_B = [ind for ind in marginds_list[indexB] if ind < 4]
        bias_marginds_B = np.argsort(bias_marginds_B)
        left_marginds_x = [ind for ind in marginds_list[indexx] if ind >= 8]
        left_marginds_x = np.argsort(left_marginds_x)
        self.move = {
            'A': {'from': bias_marginds_A, 'to': bias_marginds_A},
            'B': {
                'from': bias_marginds_B + len(bias_marginds_A),
                'to': bias_marginds_B
            },
            'x': left_marginds_x + len(bias_marginds_A) + len(bias_marginds_B),
        }
        self.ng = len(prior)
        # TODO: support one or two components are not marginalized
        if not all(self.marginds_list):
            raise ValueError(
                f"each component should have at least one marginalized paramter"
            )

    def PG(self, all_params_dict: Dict[str, Any]) -> NDArray:
        # TODO: this implementation is ugly, should be improved
        nvec = [
            np.prod(theory.bird.PG.shape[1:]) for theory in self.theories]  # type: ignore
        pad_widths = [
            ((0, 0), (0, nvec[1] + nvec[2])),
            ((0, 0), (nvec[0], nvec[2])),
            ((0, 0), (nvec[0] + nvec[1], 0))
        ]
        tmp = [
            np.vstack([theory.bird.PG[i, ...].reshape(-1)  # type: ignore
                      for i in inds])
            for theory, inds in zip(self.theories, self.marginds_list)
        ]
        tmp = [
            np.pad(x, pad) * coef[:, None]
            for x, pad, coef in zip(tmp, pad_widths, self.margcoef_list)
        ]
        indexA, indexB, indexx = [
            self._type_to_index[t] for t in ('A', 'B', 'x')]
        move = self.move
        arrA, arrB, arrx = tmp[indexA], tmp[indexB], tmp[indexx]
        arrA[move['A']['to'], :] += arrx[move['A']['from'], :]
        arrB[move['B']['to'], :] += arrx[move['B']['from'], :]
        arrx = arrx[move['x']]
        out: List[Any] = [None, None, None]
        out[indexA] = arrA
        out[indexB] = arrB
        out[indexx] = arrx
        return np.vstack(out)

    def PNG(self, all_params_dict: Dict[str, Any]) -> NDArray:
        return self.theory_vector(all_params_dict)



class CrossEFT:
    theory: EFTTheory
    prefix: str

    def __init__(self, theory: EFTTheory, prefix: str = "") -> None:
        self.theory = theory
        self.prefix = prefix
        self._set_required_params()
        self.can_marg = True

    def set_provider(self, provider: Provider) -> None:
        self.theory.set_boltzmann_provider(
            CobayaCambProvider(provider, self.theory.z)
        )

    def set_camb_provider(self, **kwargs) -> None:
        self.theory.set_boltzmann_provider(
            CambProvider(z=self.theory.z, **kwargs))

    def required_params(self) -> Dict[str, Any]:
        return self._required_params

    def _set_required_params(self) -> None:
        z = self.theory.z
        extra_zs = [] if z == 0. else [0.]
        requires = {
            'Pk_grid': {
                'nonlinear': False,
                'z': [z],
                'k_max': 5
            },
            'Hubble': {'z': extra_zs + [z]},
            'angular_diameter_distance': {'z': [z]},
            'fsigma8': {'z': [z]},
            'sigma8_z': {'z': [z]},
            'rdrag': None
        }
        
        eft_params = [
            self.prefix + name for name in
            ('b1A', 'b2A', 'b3A', 'b4A',
             'cctA', 'cr1A', 'cr2A',
             'b1B', 'b2B', 'b3B', 'b4B',
             'cctB', 'cr1B', 'cr2B',
             'ce0x', 'cemonox', 'cequadx')
        ]
        eft_requires = dict(
            zip(eft_params, [None for _ in range(len(eft_params))])
        )
        requires.update(eft_requires)
        self._required_params = requires

    def theory_vector(self, all_params_dict: Dict[str, Any]) -> NDArray:
        prefix = self.prefix
        (
            b1A, b2A, b3A, b4A,
            cctA, cr1A, cr2A,
            b1B, b2B, b3B, b4B,
            cctB, cr1B, cr2B,
            ce0x, cemonox, cequadx,
        ) = [all_params_dict[prefix + name] for name in (
            'b1A', 'b2A', 'b3A', 'b4A',
            'cctA', 'cr1A', 'cr2A',
            'b1B', 'b2B', 'b3B', 'b4B',
            'cctB', 'cr1B', 'cr2B',
            'ce0x', 'cemonox', 'cequadx',
        )]
        bsA = [b1A, b2A, b3A, b4A, cctA, cr1A, cr2A]
        bsB = [b1B, b2B, b3B, b4B, cctB, cr1B, cr2B]
        esx = [ce0x, cemonox, cequadx]
        return self.theory.theory_vector(bsA, bsB=bsB, es=esx)

    def can_marg_params(self) -> List[str]:
        return [
            self.prefix + x
            for x in ('b3A', 'cctA', 'cr1A', 'cr2A', 'b3B', 'cctB', 'cr1B', 'cr2B','ce0x', 'cemonox', 'cequadx')
        ]

    def set_marg(self, prior: Dict[str, Any]) -> None:
        self.theory.marg = True
        all_params = self.can_marg_params()
        marginds = [all_params.index(name) for name in prior.keys()]
        self.marginds = marginds


    def PG(self, all_params_dict: Dict[str, Any]) -> NDArray:
        out = np.vstack(
            [self.theory.bird.PG[i, ...].reshape(-1) for i in self.marginds])  # type: ignore
        return out

    def PNG(self, all_params_dict: Dict[str, Any]) -> NDArray:
        return self.theory_vector(all_params_dict)
