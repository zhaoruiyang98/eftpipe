# global
import sys
from copy import deepcopy
from typing import (
    Dict,
    Any,
    overload,
    Type,
)
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
# local
from eftpipe.lssdata import FullShapeData, PklData
from eftpipe.theory import (
    EFTTheory,
    SingleTracerEFT,
    TwoTracerEFT,
    TwoTracerCrossEFT,
)
from eftpipe.marginal import MargGaussian
from eftpipe.typing import LogFunc


class FullShapeDataParser:
    """a factory to create FullShapeData object

    Parameters
    ----------
    dct: dict[str, Any]
        a dictionary which contains all the information to create FullShapeData
    logfunc: Callable[[str], None]
        function used for logging, default print

    Methods
    -------
    helper_dict(cls):
        a dict template
    create_gaussian_data(self):
        create a FullShapeData object
    """

    def __init__(
        self,
        dct: Dict[str, Any],
        logfunc: LogFunc = print,
    ) -> None:
        dct = deepcopy(dct)
        pklinfo = dct.pop('pklinfo')
        if not isinstance(pklinfo, list):
            pklinfo = [pklinfo]

        common = dct.pop('common', None)
        if common is not None:
            new_pklinfo = [deepcopy(common) for _ in pklinfo]
            for raw, new in zip(new_pklinfo, pklinfo):
                raw.update(new)
            pklinfo = new_pklinfo

        self._pklinfo = pklinfo
        dct.pop('pkldatas', None)
        self.dct = dct
        self.logfunc = logfunc

    @classmethod
    def helper_dict(cls):
        return {
            "cov_path": "",
            "Nreal": 1000,
            "rescale": 1.0,
            "common": {"kmin": 0.02, "kmax": 0.2, "ls": [0, 2]},
            "pklinfo": [
                {"pkl_path": ""},
                {"pkl_path": ""},
            ]
        }

    def create_gaussian_data(self, quiet=False) -> FullShapeData:
        logfunc = self.logfunc
        if quiet:
            logfunc = lambda t: None
        pkldatas = [PklData(**x, logfunc=logfunc) for x in self._pklinfo]
        return FullShapeData(pkldatas=pkldatas, logfunc=logfunc, **self.dct)


class SingleTracerParser:
    """a factory to create SingleTracerEFT object

    Parameters
    ----------
    dct: dict[str, Any]
        a dictionary which contains all the information to create SingleTracerEFT
    logfunc: Callable[[str], None]
        function used for logging, default print

    Methods
    -------
    helper_dict(cls):
        a dict template
    create_gaussian_data(self):
        create a FullShapeData object
    create_vector_theory(self):
        create a SingleTracerEFT object
    """

    def __init__(
        self,
        dct: Dict[str, Any],
        logfunc=print
    ) -> None:
        self._data_parser = FullShapeDataParser(dct['data'], logfunc=logfunc)
        theory_info = deepcopy(dct['theory'])
        theory_info['projection_config']['kdata'] = None
        theory_info['print_info'] = logfunc
        prefix = str(theory_info.pop('prefix', ""))
        self._prefix = prefix
        self._theory_info = theory_info
        self.logfunc = logfunc
        self.marg_info = deepcopy(dct.get('marg', {}))

    def create_gaussian_data(self, quiet=False) -> FullShapeData:
        out = self._data_parser.create_gaussian_data(quiet=quiet)
        if len(out.pkldatas) != 1:
            raise ValueError('SingleTracerParser only accept one data')
        return out

    def create_vector_theory(self) -> SingleTracerEFT:
        data_obj = self.create_gaussian_data(quiet=True)
        kdata = data_obj.pkldatas[0].kdata
        self._theory_info['projection_config']['kdata'] = kdata
        theory = EFTTheory(**self._theory_info)
        return SingleTracerEFT(theory, self._prefix)

    def create_marglike(self, data_obj, vector_theory):
        return MargGaussian(data_obj, vector_theory, self.marg_info, self.logfunc)

    @classmethod
    def helper_dict(cls):
        return {
            "data": {
                "cov_path": "",
                "Nreal": 1000,
                "rescale": 1.0,
                "pklinfo": {
                    "ls": [0, 2],
                    "kmin": 0.02,
                    "kmax": 0.2,
                    "pkl_path": "",
                }
            },
            "theory": {
                "prefix": "",
                "Nl": 2,
                "z": 0.5,
                "kmA": 0.7,
                "ndA": 7.91e-05,
                "cache_dir_path": "",
                "chained": False,
                "projection_config": {
                    "Om_AP": 0.3,
                    "z_AP": 0.5,
                    "rdrag_fid": 150.0,
                    "window_fourier_path": "",
                    "window_configspace_path": "",
                    "binning": True,
                }
            }
        }


class TwoTracerParser:
    """a factory to create TwoTracerEFT object

    Parameters
    ----------
    dct: dict[str, Any]
        a dictionary which contains all the information to create TwoTracerEFT
    logfunc: Callable[[str], None]
        function used for logging, default print

    Methods
    -------
    helper_dict(cls):
        a dict template
    create_gaussian_data(self):
        create a FullShapeData object
    create_vector_theory(self):
        create a TwoTracerEFT object
    """

    def __init__(
        self,
        dct: Dict[str, Any],
        logfunc=print
    ) -> None:
        self._data_parser = FullShapeDataParser(dct['data'], logfunc=logfunc)
        theory_info = deepcopy(dct['theory'])
        prefixes = theory_info.pop('prefix')
        if (not isinstance(prefixes, list)) or (len(prefixes) != 2):
            raise ValueError('prefix should be a list with 2 items')
        prefixes = [str(item) for item in prefixes]
        self._prefixes = prefixes

        theory_infos = theory_info['theory_info']
        common = theory_info.pop('common', None)
        if common is not None:
            new_theory_infos = [deepcopy(common) for _ in theory_infos]
            for raw, new in zip(new_theory_infos, theory_infos):
                raw_projection = raw.pop('projection_config', {})
                new_projection = new.pop('projection_config', {})
                raw.update(new)
                raw_projection.update(new_projection)
                raw['projection_config'] = raw_projection
            theory_infos = new_theory_infos
        self._theory_infos = theory_infos
        self.logfunc = logfunc
        self.marg_info = deepcopy(dct.get('marg', {}))

    def create_gaussian_data(self, quiet=False) -> FullShapeData:
        out = self._data_parser.create_gaussian_data(quiet=quiet)
        if len(out.pkldatas) != 2:
            raise ValueError('TwoTracerParser only accept two data configs')
        return out

    def create_vector_theory(self) -> TwoTracerEFT:
        data_obj = self.create_gaussian_data(quiet=True)
        for theory_info, pkldata in zip(self._theory_infos, data_obj.pkldatas):
            theory_info['projection_config']['kdata'] = pkldata.kdata
        theories = [
            EFTTheory(**theory_info, print_info=self.logfunc)
            for theory_info in self._theory_infos
        ]
        return TwoTracerEFT(theories, self._prefixes)

    def create_marglike(self, data_obj, vector_theory):
        return MargGaussian(data_obj, vector_theory, self.marg_info, self.logfunc)

    @classmethod
    def helper_dict(cls):
        return {
            "data": {
                "cov_path": "",
                "Nreal": 1000,
                "rescale": 1.0,
                "pklinfo": [
                    {"pkl_path": ""},
                    {"pkl_path": ""},
                ],
                "common": {
                    "ls": [0, 2],
                    "kmin": 0.02,
                    "kmax": 0.15,
                }
            },
            "theory": {
                "prefix": ["", ""],
                "theory_info": [
                    {
                        "z": 0.7,
                        "kmA": 0.7,
                        "ndA": 7.91e-05,
                        "projection_config": {
                            "z_AP": 0.7,
                            "window_fourier_path": "",
                            "window_configspace_path": "",
                        }
                    },
                    {
                        "z": 0.77,
                        "kmA": 0.45,
                        "ndA": 0.00018518518518518518,
                        "projection_config": {
                            "z_AP": 0.77,
                            "window_fourier_path": "",
                            "window_configspace_path": "",
                        }
                    },
                ],
                "common": {
                    "cache_dir_path": "",
                    "Nl": 2,
                    "chained": False,
                    "projection_config": {
                        "Om_AP": 0.3,
                        "rdrag_fid": 150.0,
                        "binning": True
                    }
                }
            }
        }


class TwoTracerCrossParser:
    """a factory to create TwoTracerCrossEFT object

    Parameters
    ----------
    dct: dict[str, Any]
        a dictionary which contains all the information to create TwoTracerCrossEFT
    logfunc: Callable[[str], None]
        function used for logging, default print

    Methods
    -------
    helper_dict(cls):
        a dict template
    create_gaussian_data(self):
        create a FullShapeData object
    create_vector_theory(self):
        create a TwoTracerEFT object
    """

    def __init__(
        self,
        dct: Dict[str, Any],
        logfunc=print
    ) -> None:
        self._data_parser = FullShapeDataParser(dct['data'], logfunc=logfunc)
        theory_info = deepcopy(dct['theory'])
        prefixes = theory_info.pop('prefix')
        if (not isinstance(prefixes, list)) or (len(prefixes) != 3):
            raise ValueError('prefix should be a list with 3 items')
        prefixes = [str(item) for item in prefixes]
        self._prefixes = prefixes

        theory_infos = theory_info['theory_info']
        common = theory_info.pop('common', None)
        if common is not None:
            new_theory_infos = [deepcopy(common) for _ in theory_infos]
            for raw, new in zip(new_theory_infos, theory_infos):
                raw_projection = raw.pop('projection_config', {})
                new_projection = new.pop('projection_config', {})
                raw.update(new)
                raw_projection.update(new_projection)
                raw['projection_config'] = raw_projection
            theory_infos = new_theory_infos
        # TODO: cross's km, nd can be set from other two theories
        self._theory_infos = theory_infos
        self.logfunc = logfunc
        self.marg_info = deepcopy(dct.get('marg', {}))

    def create_gaussian_data(self, quiet=False) -> FullShapeData:
        out = self._data_parser.create_gaussian_data(quiet=quiet)
        if len(out.pkldatas) != 3:
            raise ValueError(
                'TwoTracerCrossParser only accept three data configs')
        return out

    def create_vector_theory(self) -> TwoTracerCrossEFT:
        data_obj = self.create_gaussian_data(quiet=True)
        for theory_info, pkldata in zip(self._theory_infos, data_obj.pkldatas):
            theory_info['projection_config']['kdata'] = pkldata.kdata
        theories = [
            EFTTheory(**theory_info, print_info=self.logfunc)
            for theory_info in self._theory_infos
        ]
        return TwoTracerCrossEFT(theories, self._prefixes)

    def create_marglike(self, data_obj, vector_theory):
        return MargGaussian(data_obj, vector_theory, self.marg_info, self.logfunc)

    @classmethod
    def helper_dict(cls):
        return {
            "data": {
                "cov_path": "",
                "Nreal": 1000,
                "rescale": 1.0,
                "pklinfo": [
                    {"pkl_path": ""},
                    {"pkl_path": ""},
                    {"pkl_path": ""},
                ],
                "common": {
                    "ls": [0, 2],
                    "kmin": 0.02,
                    "kmax": 0.15,
                }
            },
            "theory": {
                "prefix": ["", "", ""],
                "theory_info": [
                    {
                        "z": 0.7,
                        "kmA": 0.7,
                        "ndA": 7.91e-05,
                        "projection_config": {
                            "z_AP": 0.7,
                            "window_fourier_path": "",
                            "window_configspace_path": "",
                        }
                    },
                    {
                        "z": 0.845,
                        "kmA": 0.45,
                        "ndA": 0.00018518518518518518,
                        "projection_config": {
                            "z_AP": 0.845,
                            "window_fourier_path": "",
                            "window_configspace_path": "",
                        }
                    },
                    {
                        "z": 0.77,
                        "kmA": 0.7,
                        "ndA": 7.91e-05,
                        "kmB": 0.45,
                        "ndB": 0.00018518518518518518,
                        "cross": True,
                        "projection_config": {
                            "z_AP": 0.77,
                            "window_fourier_path": "",
                            "window_configspace_path": "",
                        }
                    },
                ],
                "common": {
                    "cache_dir_path": "",
                    "Nl": 2,
                    "chained": False,
                    "projection_config": {
                        "Om_AP": 0.3,
                        "rdrag_fid": 150.0,
                        "binning": True
                    }
                }
            }
        }


@overload
def select_parser(mode: Literal['single']) -> Type[SingleTracerParser]: ...
@overload
def select_parser(mode: Literal['two']) -> Type[TwoTracerParser]: ...
@overload
def select_parser(mode: Literal['all']) -> Type[TwoTracerCrossParser]: ...


def select_parser(mode: str):
    if mode == 'single':
        return SingleTracerParser
    elif mode == 'two':
        return TwoTracerParser
    elif mode == 'all':
        return TwoTracerCrossParser
    else:
        raise ValueError(f"unexpected mode {mode}")
