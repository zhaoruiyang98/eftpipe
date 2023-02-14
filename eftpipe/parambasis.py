from __future__ import annotations
import importlib
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Literal, Protocol, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Container
    from .binning import Binning
    from .pybird.pybird import Bird


class Everything:
    def __contains__(self, item: Any) -> bool:
        return True


EVERYTHING = Everything()


class EFTBasis(Protocol):
    def __init__(self, prefix: str, cross_prefix: list[str]):
        ...

    @classmethod
    def get_name(cls) -> str:
        return cls.__class__.__name__

    @classmethod
    def counterform(cls) -> Literal["westcoast", "eastcoast"]:
        ...

    def non_gaussian_params(self) -> list[str]:
        ...

    def gaussian_params(self) -> list[str]:
        ...

    def reduce_Pk(self, bird: Bird, params_values_dict: dict[str, float]) -> None:
        ...

    def create_PG_table(
        self,
        bird: Bird,
        params_values_dict: dict[str, float],
        requires: Container[str] | None = None,
    ) -> dict[str, Any]:
        ...

    def create_binned_PG_table(
        self,
        binning: Binning,
        bird: Bird,
        params_values_dict: dict[str, float],
        requires: Container[str] | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class WestCoastBasis(EFTBasis):
    prefix: str = ""
    cross_prefix: list[str] = field(default_factory=list)

    def default(self) -> dict[str, float]:
        return {p: 0.0 for p in self.gaussian_params()}

    def bsA(self) -> list[str]:
        names = ("b1", "b2", "b3", "b4", "cct", "cr1", "cr2")
        prefix = self.cross_prefix[0] if self.is_cross() else self.prefix
        return [prefix + p for p in names]

    def bsB(self) -> list[str]:
        if not self.is_cross():
            return []
        names = ("b1", "b2", "b3", "b4", "cct", "cr1", "cr2")
        prefix = self.cross_prefix[1]
        return [prefix + p for p in names]

    def es(self) -> list[str]:
        return [self.prefix + p for p in ("ce0", "cemono", "cequad")]

    def is_cross(self) -> bool:
        return True if self.cross_prefix else False

    # override
    @classmethod
    def get_name(cls) -> str:
        return "westcoast"

    # impl
    @classmethod
    def counterform(cls):
        return "westcoast"

    # impl
    def non_gaussian_params(self) -> list[str]:
        names = ("b1", "b2", "b4")
        if self.is_cross():
            retval = [x + p for x in self.cross_prefix for p in names]
        else:
            retval = [self.prefix + p for p in names]
        return retval

    # impl
    def gaussian_params(self) -> list[str]:
        names = ("b3", "cct", "cr1", "cr2")
        stnames = ("ce0", "cemono", "cequad")
        if self.is_cross():
            retval = [x + p for x in self.cross_prefix for p in names]
            retval += [self.prefix + p for p in stnames]
        else:
            retval = [self.prefix + p for p in names + stnames]
        return retval

    # impl
    def reduce_Pk(self, bird: Bird, params_values_dict: dict[str, float]) -> None:
        param_values = self.default()
        param_values.update(params_values_dict)
        bsA = [param_values[p] for p in self.bsA()]
        bsB = [param_values[p] for p in self.bsB()] or None
        es = [param_values[p] for p in self.es()]
        bird.setreducePslb(bsA, bsB, es)

    # impl
    def create_PG_table(
        self,
        bird: Bird,
        params_values_dict: dict[str, float],
        requires: Container[str] | None = None,
    ) -> dict[str, Any]:
        f = bird.f
        kmA, krA, ndA, kmB, krB, ndB = (
            bird.co.kmA,
            bird.co.krA,
            bird.co.ndA,
            bird.co.kmB,
            bird.co.krB,
            bird.co.ndB,
        )
        if self.is_cross():
            b1A, b1B = (params_values_dict[_ + "b1"] for _ in self.cross_prefix)
        else:
            b1A = b1B = params_values_dict[self.prefix + "b1"]
        Ploopl, Pctl, Pstl = bird.Ploopl, bird.Pctl, bird.Pstl
        return self.create_PG_table_subroutine(
            f, kmA, krA, ndA, kmB, krB, ndB, b1A, b1B, Ploopl, Pctl, Pstl, requires
        )

    # impl
    def create_binned_PG_table(
        self,
        binning: Binning,
        bird: Bird,
        params_values_dict: dict[str, float],
        requires: Container[str] | None = None,
    ) -> dict[str, Any]:
        f = bird.f
        kmA, krA, ndA, kmB, krB, ndB = (
            binning.co.kmA,
            binning.co.krA,
            binning.co.ndA,
            binning.co.kmB,
            binning.co.krB,
            binning.co.ndB,
        )
        if self.is_cross():
            b1A, b1B = (params_values_dict[_ + "b1"] for _ in self.cross_prefix)
        else:
            b1A = b1B = params_values_dict[self.prefix + "b1"]
        Ploopl = binning.Ploopl
        Pctl = binning.Pctl
        Pstl = binning.Pstl
        return self.create_PG_table_subroutine(
            f, kmA, krA, ndA, kmB, krB, ndB, b1A, b1B, Ploopl, Pctl, Pstl, requires
        )

    def create_PG_table_subroutine(
        self,
        f: float,
        kmA: float,
        krA: float,
        ndA: float,
        kmB: float,
        krB: float,
        ndB: float,
        b1A: float,
        b1B: float,
        Ploopl: Any,
        Pctl: Any,
        Pstl: Any,
        requires: Container[str] | None = None,
    ):
        if requires is None:
            requires = EVERYTHING
        PG: dict[str, Any] = {}
        if self.is_cross():
            _A, _B = self.cross_prefix
            if (p := _A + "b3") in requires:
                PG[p] = 1 / 2 * Ploopl[:, 3, :] + 1 / 2 * b1B * Ploopl[:, 7, :]
            if (p := _A + "cct") in requires:
                PG[p] = b1B / kmA**2 * Pctl[:, 0, :] + f / kmA**2 * Pctl[:, 3, :]
            if (p := _A + "cr1") in requires:
                PG[p] = b1B / krA**2 * Pctl[:, 1, :] + f / krA**2 * Pctl[:, 4, :]
            if (p := _A + "cr2") in requires:
                PG[p] = b1B / krA**2 * Pctl[:, 2, :] + f / krA**2 * Pctl[:, 5, :]
            if (p := _B + "b3") in requires:
                PG[p] = 1 / 2 * Ploopl[:, 3, :] + 1 / 2 * b1A * Ploopl[:, 7, :]
            if (p := _B + "cct") in requires:
                PG[p] = b1A / kmB**2 * Pctl[:, 0, :] + f / kmB**2 * Pctl[:, 3, :]
            if (p := _B + "cr1") in requires:
                PG[p] = b1A / krB**2 * Pctl[:, 1, :] + f / krB**2 * Pctl[:, 4, :]
            if (p := _B + "cr2") in requires:
                PG[p] = b1A / krB**2 * Pctl[:, 2, :] + f / krB**2 * Pctl[:, 5, :]
        else:
            if (p := self.prefix + "b3") in requires:
                PG[p] = Ploopl[:, 3, :] + b1A * Ploopl[:, 7, :]
            if (p := self.prefix + "cct") in requires:
                PG[p] = (
                    2.0 * b1A / kmA**2 * Pctl[:, 0, :]
                    + 2.0 * f / kmA**2 * Pctl[:, 3, :]
                )
            if (p := self.prefix + "cr1") in requires:
                PG[p] = (
                    2.0 * b1A / krA**2 * Pctl[:, 1, :]
                    + 2.0 * f / krA**2 * Pctl[:, 4, :]
                )
            if (p := self.prefix + "cr2") in requires:
                PG[p] = (
                    2.0 * b1A / krA**2 * Pctl[:, 2, :]
                    + 2.0 * f / krA**2 * Pctl[:, 5, :]
                )
        xfactor1 = 0.5 * (1.0 / ndA + 1.0 / ndB)
        xfactor2 = 0.5 * (1.0 / ndA / kmA**2 + 1.0 / ndB / kmB**2)
        if (p := self.prefix + "ce0") in requires:
            PG[p] = Pstl[:, 0, :] * xfactor1
        if (p := self.prefix + "cemono") in requires:
            PG[p] = Pstl[:, 1, :] * xfactor2
        if (p := self.prefix + "cequad") in requires:
            PG[p] = Pstl[:, 2, :] * xfactor2
        return PG


def find_param_basis(name: str) -> Type[EFTBasis]:
    if name == "westcoast":
        return WestCoastBasis
    else:
        module_name, class_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
