# global
import sys
import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.theory import Provider
from copy import deepcopy
from pathlib import Path
from numpy import ndarray as NDArray
from typing import (
    Optional,
    List,
    Union,
    Dict,
    Any,
    Tuple,
    Callable,
    cast,
)
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
# local
from eftpipe.lssdata import FullShapeData
from eftpipe.theory import EFTTheory
from eftpipe.interface import CobayaCambProvider
from eftpipe.typing import (
    Location,
    GaussianData,
    VectorTheory,
)


class EFTLike(Likelihood):
    label: str
    nsampled: int
    extra_args: Dict[str, Any]
    # intermediate products
    input_params: Dict[str, Any]
    data_obj: GaussianData
    theory_obj: VectorTheory

    def initialize(self) -> None:
        data_obj, theory_obj = construct_data_and_theory_obj(
            self.extra_args, self.mpi_info)
        self.data_obj = data_obj
        self.theory_obj = theory_obj
        if self.nsampled is None:
            raise ValueError(
                f"please specify the number of sampled parameters")
        if self.label is None:
            raise ValueError(f"please specify the label of likelihood")
        self.label = str(self.label)

    def initialize_with_provider(self, provider: 'Provider'):
        super().initialize_with_provider(provider)
        self.theory_obj.set_provider(self.provider)

    def get_requirements(self) -> Dict[str, Any]:
        return self.theory_obj.required_params()

    def calculate(self, state, want_derived=True, **params_values_dict):
        theory = self.theory_obj.theory_vector(params_values_dict)
        res = theory - self.data_obj.data_vector
        chi2 = res @ self.data_obj.invcov @ res

        if want_derived:
            state['derived'] = {
                self.label + 'reduced_chi2':
                chi2 / (self.data_obj.ndata - self.nsampled)
            }
        state['logp'] = -0.5 * chi2

    def get_can_provide_params(self) -> List[str]:
        return [self.label + 'reduced_chi2']


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
             'ce0', 'cemono', 'cequad',
             'knl', 'km', 'nd')
        ]
        eft_requires = dict(
            zip(eft_params, [None for _ in range(len(eft_params))])
        )
        requires.update(eft_requires)
        self._required_params = requires

    def theory_vector(self, all_params_dict: Dict[str, Any]) -> NDArray:
        rdrag_true = all_params_dict.get('rdrag', None)
        prefix = self.prefix
        (
            b1, b2, b3, b4,
            cct, cr1, cr2,
            ce0, cemono, cequad,
            knl, km, nd,
        ) = [all_params_dict[prefix + name] for name in (
            'b1', 'b2', 'b3', 'b4',
            'cct', 'cr1', 'cr2',
            'ce0', 'cemono', 'cequad',
            'knl', 'km', 'nd',
        )]
        bs = [
            b1, b2, b3, b4, cct/knl**2, cr1/km**2, cr2/km**2,
        ]
        es = [
            ce0/nd, cemono/nd/km**2, cequad/nd/km**2,
        ]
        return self.theory.theory_vector(bs, es=es, rdrag_true=rdrag_true)


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
            'knl', 'km', 'nd'
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
        rdrag_true = all_params_dict.get('rdrag', None)
        vectors = []
        for (prefix, theory) in zip(self.prefixes, self.theories):
            (
                b1, b2, b3, b4,
                cct, cr1, cr2,
                ce0, cemono, cequad,
                knl, km, nd,
            ) = [all_params_dict[prefix + name] for name in (
                'b1', 'b2', 'b3', 'b4',
                'cct', 'cr1', 'cr2',
                'ce0', 'cemono', 'cequad',
                'knl', 'km', 'nd',
            )]
            bs = [
                b1, b2, b3, b4, cct/knl**2, cr1/km**2, cr2/km**2,
            ]
            es = [
                ce0/nd, cemono/nd/km**2, cequad/nd/km**2,
            ]
            vectors.append(theory.theory_vector(
                bs, es=es, rdrag_true=rdrag_true))
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
            'knl', 'km', 'nd'
        ]
        cross_params_names = ['ce0', 'cemono', 'cequad', 'km']
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
        rdrag_true = all_params_dict.get('rdrag', None)
        eft_params_names = [
            'b1', 'b2', 'b3', 'b4',
            'cct', 'cr1', 'cr2',
            'ce0', 'cemono', 'cequad',
            'knl', 'km', 'nd'
        ]
        cross_params_names = ['ce0', 'cemono', 'cequad', 'km']
        Aindex, Bindex, xindex = [
            self._index_mapping[key] for key in ('A', 'B', 'x')]
        prefixA = self.prefixes[Aindex]
        prefixB = self.prefixes[Bindex]
        prefixx = self.prefixes[xindex]
        (
            b1A, b2A, b3A, b4A,
            cctA, cr1A, cr2A,
            ce0A, cemonoA, cequadA,
            knlA, kmA, ndA
        ) = [all_params_dict[prefixA + name] for name in eft_params_names]
        bsA = [b1A, b2A, b3A, b4A, cctA/knlA, cr1A/kmA**2, cr2A/kmA**2]
        esA = [ce0A/ndA, cemonoA/ndA/kmA**2, cequadA/ndA/kmA**2]
        (
            b1B, b2B, b3B, b4B,
            cctB, cr1B, cr2B,
            ce0B, cemonoB, cequadB,
            knlB, kmB, ndB
        ) = [all_params_dict[prefixB + name] for name in eft_params_names]
        bsB = [b1B, b2B, b3B, b4B, cctB/knlB, cr1B/kmB**2, cr2B/kmB**2]
        esB = [ce0B/ndB, cemonoB/ndB/kmB**2, cequadB/ndB/kmB**2]
        ce0x, cemonox, cequadx, kmx = [
            all_params_dict[prefixx + name] for name in cross_params_names]
        nfactorx = 0.5 * (1./ndA + 1./ndB)
        esx = [
            nfactorx * ce0x, nfactorx * cemonox/kmx**2,
            nfactorx * cequadx/kmx**2]

        theory_vectorA = self.theories[Aindex].theory_vector(
            bsA, es=esA, rdrag_true=rdrag_true)
        theory_vectorB = self.theories[Bindex].theory_vector(
            bsB, es=esB, rdrag_true=rdrag_true)
        theory_vectorx = self.theories[xindex].theory_vector(
            bsA, bsB=bsB, es=esx, rdrag_true=rdrag_true)
        vectors = cast(List[NDArray], [None, None, None])
        vectors[Aindex] = theory_vectorA
        vectors[Bindex] = theory_vectorB
        vectors[xindex] = theory_vectorx
        return np.hstack(vectors)


def construct_data_and_theory_obj(
    info: Dict[str, Any],
    log_func: Optional[Callable] = None
):
    mode = info.get('mode')
    if mode is None:
        raise KeyError("missing 'mode' in extra_args")
    elif mode == 'single':
        out = parse_single_tracer(info, log_func)
    elif mode == 'two':
        out = parse_two_tracer(info, log_func)
    elif mode == 'all':
        out = parse_all_tracer(info, log_func)
    else:
        raise ValueError(f"unexpected mode {mode}")
    return out


def parse_single_tracer(
    info: Dict[str, Any],
    log_func: Optional[Callable] = None
) -> Tuple[FullShapeData, SingleTracerEFT]:
    new_info = deepcopy(info)
    data_info = new_info['data']
    theory_info: Dict[str, Any] = new_info['theory']
    update_path_in_dict(data_info)
    update_path_in_dict(theory_info)
    data_obj = FullShapeData(**data_info)
    prefix = str(theory_info.pop('prefix', ""))
    theory_info['projection_config']['kdata'] = data_obj.pkldatas[0].kdata
    if log_func is not None:
        theory_info['print_info'] = log_func
    theory = EFTTheory(**theory_info)
    theory_obj = SingleTracerEFT(theory, prefix)
    return data_obj, theory_obj


def parse_two_tracer(
    info: Dict[str, Any],
    log_func: Optional[Callable] = None
) -> Tuple[FullShapeData, TwoTracerEFT]:
    new_info = deepcopy(info)
    data_info = new_info['data']
    theory_info: Dict[str, Any] = new_info['theory']
    update_path_in_dict(data_info)
    update_path_in_dict(theory_info)
    data_obj = FullShapeData(**data_info)

    prefixes = theory_info.pop('prefix', None)
    if (not isinstance(prefixes, list)) or (len(prefixes) != 2):
        raise ValueError('prefix should be a list with 2 items')
    prefixes = [str(item) for item in prefixes]
    common = theory_info.pop('common', {})
    theories: List[EFTTheory] = []
    for pkldata, info in zip(data_obj.pkldatas, theory_info['theory_info']):
        new_common = deepcopy(common)
        if 'projection_config' not in new_common:
            new_common['projection_config'] = {}
        for key, value in info.items():
            if key != 'projection_config':
                new_common[key] = value
            else:
                for proj_key, proj_value in value.items():
                    new_common['projection_config'][proj_key] = proj_value
        new_common['projection_config']['kdata'] = pkldata.kdata
        if log_func is not None:
            new_common['print_info'] = log_func
        theory = EFTTheory(**new_common)
        theories.append(theory)
    theory_obj = TwoTracerEFT(theories, prefixes)
    return data_obj, theory_obj


def parse_all_tracer(
    info: Dict[str, Any],
    log_func: Optional[Callable] = None
) -> Tuple[FullShapeData, TwoTracerCrossEFT]:
    new_info = deepcopy(info)
    data_info = new_info['data']
    theory_info: Dict[str, Any] = new_info['theory']
    update_path_in_dict(data_info)
    update_path_in_dict(theory_info)
    data_obj = FullShapeData(**data_info)

    prefixes = theory_info.pop('prefix', None)
    if (not isinstance(prefixes, list)) or (len(prefixes) != 3):
        raise ValueError('prefix should be a list with 3 items')
    prefixes = [str(item) for item in prefixes]
    common = theory_info.pop('common', {})
    theories: List[EFTTheory] = []
    for pkldata, info in zip(data_obj.pkldatas, theory_info['theory_info']):
        new_common = deepcopy(common)
        if 'projection_config' not in new_common:
            new_common['projection_config'] = {}
        for key, value in info.items():
            if key != 'projection_config':
                new_common[key] = value
            else:
                for proj_key, proj_value in value.items():
                    new_common['projection_config'][proj_key] = proj_value
        new_common['projection_config']['kdata'] = pkldata.kdata
        if log_func is not None:
            new_common['print_info'] = log_func
        theory = EFTTheory(**new_common)
        theories.append(theory)
    theory_obj = TwoTracerCrossEFT(theories, prefixes)
    return data_obj, theory_obj


def update_path_in_dict(d: Dict) -> None:
    for key, value in d.items():
        if 'path' in key:
            d[key] = Path(value)
        if isinstance(value, dict):
            update_path_in_dict(value)
        if isinstance(value, list):
            if isinstance(value[0], dict):
                for item in value:
                    update_path_in_dict(item)
