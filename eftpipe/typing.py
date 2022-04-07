import sys
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Union,
)
from numpy import ndarray as NDArray
if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

Location = Union[str, Path]

YamlElement = Union[str, bool, int, float, None, datetime]
SimpleYaml = Union[
    YamlElement,
    List['SimpleYaml'],
    Dict[str, 'SimpleYaml']
]


class GaussianData(Protocol):
    ndata: int
    data_vector: NDArray
    invcov: NDArray


class VectorTheory(Protocol):
    can_marg: bool
    def required_params(self) -> Dict[str, Any]: ...
    def theory_vector(self, all_params_dict: Dict[str, Any]) -> NDArray: ...
    def set_provider(self, provider) -> None: ...
    def can_marg_params(self) -> List[str]: ...
    def set_marg(self, prior: Dict[str, Any]) -> None: ...
    def PG(self, all_params_dict: Dict[str, Any]) -> NDArray: ...
    def PNG(self, all_params_dict: Dict[str, Any]) -> NDArray: ...


class MargLike(Protocol):
    def __init__(
        self, _data: GaussianData, _theory: VectorTheory, prior: Dict[str, Any]
    ) -> None: ...
    def calculate(self, all_params_dict: Dict[str, Any]) -> float: ...


class ExtraArgsParser(Protocol):
    def create_gaussian_data(self) -> GaussianData: ...
    def create_vector_theory(self) -> VectorTheory: ...
    def create_marglike(
        self, __data: GaussianData, __theory: VectorTheory) -> MargLike: ...


class BoltzmannProvider(Protocol):
    def interp_pkh(self, kh: NDArray) -> NDArray: ...
    def get_f(self, z: float) -> float: ...
    def get_DA(self, z: float) -> float: ...
    def get_H(self, z: float) -> float: ...
    def get_rdrag(self) -> float: ...
    def cosmo_updated(self) -> bool: ...
