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
from eftpipe.lssdata import FullShapeData
from eftpipe.lssdata import FullShapeDataDict
from eftpipe.marginal import MargGaussian
from eftpipe.theory import CrossEFT
from eftpipe.theory import EFTTheory
from eftpipe.theory import SingleTracerEFT
from eftpipe.theory import TwoTracerCrossEFT
from eftpipe.theory import TwoTracerEFT
from eftpipe.tools import recursively_update_dict
from eftpipe.tools import set_value_in_nested_dict


# TODO: deprecated
class FullShapeDataParser:
    """a factory to create FullShapeData object

    Parameters
    ----------
    dct: dict[str, Any]
        a dictionary which contains all the information to create FullShapeData

    Methods
    -------
    create_gaussian_data(quiet=False):
        create a FullShapeData object
    """

    def __init__(self, dct: FullShapeDataDict) -> None:
        self.dct = dct

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
        return FullShapeData.from_dict(self.dct, log=not quiet)


class SingleTracerParser:
    """a factory to create SingleTracerEFT object

    Parameters
    ----------
    dct: dict[str, Any]
        a dictionary which contains all the information to create SingleTracerEFT

    Methods
    -------
    helper_dict(cls):
        a dict template
    create_gaussian_data(self):
        create a FullShapeData object
    create_vector_theory(self):
        create a SingleTracerEFT object
    
    Notes
    -----
    kdata of data_obj will be passed to config Binning
    """

    def __init__(self, dct: Dict[str, Any]) -> None:
        self._data_parser = FullShapeDataParser(dct['data'])
        theory_info = deepcopy(dct['theory'])
        prefix = str(theory_info.pop('prefix', ""))
        self._prefix = prefix
        self._theory_info = theory_info
        self.marg_info = deepcopy(dct.get('marg', {}))

    def create_gaussian_data(self, quiet=False) -> FullShapeData:
        out = self._data_parser.create_gaussian_data(quiet=quiet)
        if len(out.pkldatas) != 1:
            raise ValueError('SingleTracerParser only accept one data')
        return out

    def create_vector_theory(self) -> SingleTracerEFT:
        data_obj = self.create_gaussian_data(quiet=True)
        kdata = data_obj.pkldatas[0].kdata
        set_value_in_nested_dict(
            self._theory_info, kdata, "config_settings", "binning", "kout")
        theory = EFTTheory(**self._theory_info)
        return SingleTracerEFT(theory, self._prefix)

    def create_marglike(self, data_obj, vector_theory):
        return MargGaussian(data_obj, vector_theory, self.marg_info)

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
                "z": 0.5,
                "cache_dir_path": "",
                "km": 0.7,
                "nd": 7.91e-05,
                "Nl": 2,
                "optiresum": False,
                "chained": False,
                "with_IRresum": True,
                "with_APeffect": True,
                "with_window": True,
                "with_fiber": False,
                "with_binning": True,
                "config_settings": {
                    "APeffect": {},
                    "window": {},
                    "fiber": {},
                    "binning": {},
                },
            }
        }


class TwoTracerParser:
    """a factory to create TwoTracerEFT object

    Parameters
    ----------
    dct: dict[str, Any]
        a dictionary which contains all the information to create TwoTracerEFT

    Methods
    -------
    helper_dict(cls):
        a dict template
    create_gaussian_data(self):
        create a FullShapeData object
    create_vector_theory(self):
        create a TwoTracerEFT object

    Notes
    -----
    kdata of data_obj will be passed to config Binning
    """

    def __init__(self, dct: Dict[str, Any]) -> None:
        self._data_parser = FullShapeDataParser(dct['data'])
        theory_info = deepcopy(dct['theory'])
        prefixes = theory_info.pop('prefix')
        if (not isinstance(prefixes, list)) or (len(prefixes) != 2):
            raise ValueError('prefix should be a list with 2 items')
        prefixes = [str(item) for item in prefixes]
        self._prefixes = prefixes

        theory_infos = theory_info['theory_info']
        common = theory_info.pop('common', None)
        if common is not None:
            common_infos = [deepcopy(common) for _ in theory_infos]
            for ref, new in zip(common_infos, theory_infos):
                recursively_update_dict(ref, new)
            theory_infos = common_infos
        self._theory_infos = theory_infos
        self.marg_info = deepcopy(dct.get('marg', {}))

    def create_gaussian_data(self, quiet=False) -> FullShapeData:
        out = self._data_parser.create_gaussian_data(quiet=quiet)
        if len(out.pkldatas) != 2:
            raise ValueError('TwoTracerParser only accept two data configs')
        return out

    def create_vector_theory(self) -> TwoTracerEFT:
        data_obj = self.create_gaussian_data(quiet=True)
        for theory_info, pkldata in zip(self._theory_infos, data_obj.pkldatas):
            set_value_in_nested_dict(
                theory_info, pkldata.kdata, "config_settings", "binning", "kout")
        theories = [
            EFTTheory(**theory_info)
            for theory_info in self._theory_infos
        ]
        return TwoTracerEFT(theories, self._prefixes)

    def create_marglike(self, data_obj, vector_theory):
        return MargGaussian(data_obj, vector_theory, self.marg_info)

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
                        "km": 0.7,
                        "nd": 7.91e-05,
                        "config_settings": {
                            "APeffect": {},
                            "window": {},
                            "fiber": {},
                            "binning": {},
                        }
                    },
                    {
                        "z": 0.77,
                        "km": 0.45,
                        "nd": 0.00018518518518518518,
                        "config_settings": {
                            "APeffect": {},
                            "window": {},
                            "fiber": {},
                            "binning": {},
                        }
                    },
                ],
                "common": {
                    "cache_dir_path": "",
                    "Nl": 2,
                    "chained": False,
                    "config_settings": {
                        "APeffect": {"Om_AP": 0.3},
                        "binning": {"binning": True},
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

    Methods
    -------
    helper_dict(cls):
        a dict template
    create_gaussian_data(self):
        create a FullShapeData object
    create_vector_theory(self):
        create a TwoTracerEFT object
    
    Notes
    -----
    kdata of data_obj will be passed to config Binning
    """

    def __init__(self, dct: Dict[str, Any]) -> None:
        self._data_parser = FullShapeDataParser(dct['data'])
        theory_info = deepcopy(dct['theory'])
        prefixes = theory_info.pop('prefix')
        if (not isinstance(prefixes, list)) or (len(prefixes) != 3):
            raise ValueError('prefix should be a list with 3 items')
        prefixes = [str(item) for item in prefixes]
        self._prefixes = prefixes

        theory_infos = theory_info['theory_info']
        common = theory_info.pop('common', None)
        if common is not None:
            common_infos = [deepcopy(common) for _ in theory_infos]
            for ref, new in zip(common_infos, theory_infos):
                recursively_update_dict(ref, new)
            theory_infos = common_infos
        # TODO: cross's km, nd can be set from other two theories
        self._theory_infos = theory_infos
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
            set_value_in_nested_dict(
                theory_info, pkldata.kdata, "config_settings", "binning", "kout")
        theories = [
            EFTTheory(**theory_info)
            for theory_info in self._theory_infos
        ]
        return TwoTracerCrossEFT(theories, self._prefixes)

    def create_marglike(self, data_obj, vector_theory):
        return MargGaussian(data_obj, vector_theory, self.marg_info)

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
                        "km": 0.7,
                        "nd": 7.91e-05,
                        "config_settings": {
                            "APeffect": {},
                            "window": {},
                            "fiber": {},
                            "binning": {},
                        }
                    },
                    {
                        "z": 0.845,
                        "km": 0.45,
                        "nd": 0.00018518518518518518,
                        "config_settings": {
                            "APeffect": {},
                            "window": {},
                            "fiber": {},
                            "binning": {},
                        }
                    },
                    {
                        "z": 0.77,
                        "kmA": 0.7,
                        "ndA": 7.91e-05,
                        "kmB": 0.45,
                        "ndB": 0.00018518518518518518,
                        "cross": True,
                        "config_settings": {
                            "APeffect": {},
                            "window": {},
                            "fiber": {},
                            "binning": {},
                        }
                    },
                ],
                "common": {
                    "cache_dir_path": "",
                    "Nl": 2,
                    "chained": False,
                    "config_settings": {
                        "APeffect": {"Om_AP": 0.3},
                        "binning": {"binning": True},
                    }
                }
            }
        }

class CrossParser:
    """a factory to create CrossEFT object

    Parameters
    ----------
    dct: dict[str, Any]
        a dictionary which contains all the information to create CrossEFT

    Methods
    -------
    helper_dict(cls):
        a dict template
    create_gaussian_data(self):
        create a FullShapeData object
    create_vector_theory(self):
        create a CrossEFT object
    """

    def __init__(self, dct: Dict[str, Any]) -> None:
        self._data_parser = FullShapeDataParser(dct['data'])
        theory_info = deepcopy(dct['theory'])
        prefix = str(theory_info.pop('prefix', ""))
        self._prefix = prefix
        self._theory_info = theory_info
        self.marg_info = deepcopy(dct.get('marg', {}))

    def create_gaussian_data(self, quiet=False) -> FullShapeData:
        out = self._data_parser.create_gaussian_data(quiet=quiet)
        if len(out.pkldatas) != 1:
            raise ValueError('CrossParser only accept one data')
        return out

    def create_vector_theory(self) -> CrossEFT:
        data_obj = self.create_gaussian_data(quiet=True)
        kdata = data_obj.pkldatas[0].kdata
        if "config_settings" in self._theory_info.keys():
            if "binning" in self._theory_info["config_settings"].keys():
                self._theory_info["config_settings"]["binning"]["kout"] = kdata
        theory = EFTTheory(**self._theory_info)
        return CrossEFT(theory, self._prefix)

    def create_marglike(self, data_obj, vector_theory):
        return MargGaussian(data_obj, vector_theory, self.marg_info)

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
                "z": 0.77,
                "cache_dir_path": "",
                "kmA": 0.7,
                "ndA": 7.91e-05,
                "kmB": 0.45,
                "ndB": 0.00018518518518518518,
                "Nl": 2,
                "cross": True,
                "optiresum": False,
                "chained": False,
                "with_IRresum": True,
                "with_APeffect": True,
                "with_window": True,
                "with_fiber": False,
                "with_binning": True,
                "cross": True,
                "config_settings": {
                    "APeffect": {},
                    "window": {},
                    "fiber": {},
                    "binning": {},
                }
            }
        }


@overload
def select_parser(mode: Literal['single']) -> Type[SingleTracerParser]: ...
@overload
def select_parser(mode: Literal['two']) -> Type[TwoTracerParser]: ...
@overload
def select_parser(mode: Literal['all']) -> Type[TwoTracerCrossParser]: ...
@overload
def select_parser(mode: Literal['cross']) -> Type[CrossParser]: ...

def select_parser(mode: str):
    if mode == 'single':
        return SingleTracerParser
    elif mode == 'two':
        return TwoTracerParser
    elif mode == 'all':
        return TwoTracerCrossParser
    elif mode == 'cross':
        return CrossParser
    else:
        raise ValueError(f"unexpected mode {mode}")
