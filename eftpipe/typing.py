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
