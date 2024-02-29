from __future__ import annotations
import os
import sys
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Protocol,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    try:
        from typing_extensions import TypeAlias
    except ImportError:
        # typing_extensions is not a hard dependency
        pass

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    ndarrayf: TypeAlias = NDArray[np.float64]

FilePath: TypeAlias = Union[str, os.PathLike]
YamlElement: TypeAlias = Union[str, bool, int, float, None, datetime]
SimpleYaml: TypeAlias = Union[YamlElement, List["SimpleYaml"], Dict[str, "SimpleYaml"]]


class SupportsDunderLT(Protocol):
    def __lt__(self, __other: Any) -> bool: ...


class SupportsDunderGT(Protocol):
    def __gt__(self, __other: Any) -> bool: ...


SupportsRichComparison: TypeAlias = Union[SupportsDunderLT, SupportsDunderGT]
SupportsRichComparisonT = TypeVar("SupportsRichComparisonT", bound=SupportsRichComparison)
