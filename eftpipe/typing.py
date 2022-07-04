from __future__ import annotations
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Protocol,
    TypeVar,
    Union,
)

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from numpy.typing import NDArray

Location: TypeAlias = Union[str, Path]

YamlElement: TypeAlias = Union[str, bool, int, float, None, datetime]
SimpleYaml: TypeAlias = Union[YamlElement, List["SimpleYaml"], Dict[str, "SimpleYaml"]]


class SupportsDunderLT(Protocol):
    def __lt__(self, __other: Any) -> bool:
        ...


class SupportsDunderGT(Protocol):
    def __gt__(self, __other: Any) -> bool:
        ...


SupportsRichComparison: TypeAlias = Union[SupportsDunderLT, SupportsDunderGT]
SupportsRichComparisonT = TypeVar(
    "SupportsRichComparisonT", bound=SupportsRichComparison
)

NDArrayF8: TypeAlias = NDArray[np.float64]


class GaussianData(Protocol):
    ndata: int
    data_vector: NDArray
    invcov: NDArray


class VectorTheory(Protocol):
    can_marg: bool

    def required_params(self) -> dict[str, Any]:
        ...

    def theory_vector(self, all_params_dict: dict[str, Any]) -> NDArray:
        ...

    def set_provider(self, provider) -> None:
        ...

    def can_marg_params(self) -> list[str]:
        ...

    def set_marg(self, prior: dict[str, Any]) -> None:
        ...

    def PG(self, all_params_dict: dict[str, Any]) -> NDArray:
        ...

    def PNG(self, all_params_dict: dict[str, Any]) -> NDArray:
        ...


class MargLike(Protocol):
    def __init__(
        self, _data: GaussianData, _theory: VectorTheory, prior: dict[str, Any]
    ) -> None:
        ...

    def calculate(self, all_params_dict: dict[str, Any]) -> float:
        ...


class ExtraArgsParser(Protocol):
    def create_gaussian_data(self) -> GaussianData:
        ...

    def create_vector_theory(self) -> VectorTheory:
        ...

    def create_marglike(self, __data: GaussianData, __theory: VectorTheory) -> MargLike:
        ...


class BoltzmannProvider(Protocol):
    def interp_pkh(self, kh: NDArray) -> NDArray:
        ...

    def get_f(self, z: float) -> float:
        ...

    def get_DA(self, z: float) -> float:
        ...

    def get_H(self, z: float) -> float:
        ...

    def get_rdrag(self) -> float:
        ...

    def cosmo_updated(self) -> bool:
        ...

