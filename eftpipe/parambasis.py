from __future__ import annotations
import importlib
import numpy as np
from dataclasses import dataclass
from dataclasses import field
from typing import (
    Any,
    Container,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Type,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .pybird.pybird import BirdLike
    from .typing import ndarrayf


class Everything:
    def __contains__(self, item: Any) -> bool:
        return True


EVERYTHING = Everything()


@dataclass
class BirdComponent:
    Plin: ndarrayf
    Ploop: ndarrayf
    Pct: ndarrayf
    Pst: ndarrayf
    Picc: ndarrayf

    def sum(self) -> ndarrayf:
        return self.Plin + self.Ploop + self.Pct + self.Pst + self.Picc


def reduce_Plk(
    bird: BirdLike,
    bsA: Iterable[float],
    bsB: Iterable[float] | None = None,
    es: Iterable[float] = (0.0, 0.0, 0.0),
    cnnloA: Iterable[float] = (0.0, 0.0),
    cnnloB: Iterable[float] | None = None,
) -> BirdComponent:
    R"""compute reduced power spectrum

    Parameters
    ----------
    bsA : Iterable[float]
        b_1, b_2, b_3, b_4, c_{ct}, c_{r,1}, c{r,2}
        if counterform="eastcoast", c_{ct}, c_{r,1}, c{r,2} are interpreted as
        \tilde{c}_0, \tilde{c}_2, \tilde{c}_4
    bsB : Iterable[float], optional
        the same as bsA, but for tracer B, by default None,
        and will compute auto power spectrum
    es : Iterable[float], optional
        c_{e,0}, c_{mono}, c_{quad}, by default zeros
    cnnloA : Iterable[float]
        c_{r,4}, c_{r,6} or ctilde for eastcoast
    cnnloB : Iterable[float], optional
        the same as cnnloA, but for tracer B, by default None,
        and will compute auto power spectrum
    """
    kmA, krA, ndA, kmB, krB, ndB = (
        bird.co.kmA,
        bird.co.krA,
        bird.co.ndA,
        bird.co.kmB,
        bird.co.krB,
        bird.co.ndB,
    )
    b1A, b2A, b3A, b4A, cctA, cr1A, cr2A = bsA
    b1B, b2B, b3B, b4B, cctB, cr1B, cr2B = bsB or bsA
    f = bird.f
    ce0, cemono, cequad = es

    # cct -> cct / km**2, cr1 -> cr1 / kr**2, cr2 -> cr2 / kr**2
    # ce0 -> ce0 / nd, cemono -> cemono / nd / km**2, cequad -> cequad / nd / km**2
    b11AB = np.array([b1A * b1B, (b1A + b1B) * f, f**2])
    if bird.co.counterform == "westcoast":
        bctAB = np.array(
            [
                b1A * cctB / kmB**2 + b1B * cctA / kmA**2,
                b1B * cr1A / krA**2 + b1A * cr1B / krB**2,
                b1B * cr2A / krA**2 + b1A * cr2B / krB**2,
                (cctA / kmA**2 + cctB / kmB**2) * f,
                (cr1A / krA**2 + cr1B / krB**2) * f,
                (cr2A / krA**2 + cr2B / krB**2) * f,
            ]
        )
        if bird.co.with_NNLO:
            cr4, cr6 = cnnloA
            bctNNLOAB = np.array(
                [1 / 4 * b1A**2 / krA**4 * cr4, 1 / 4 * b1A / krA**4 * cr6, 0.0]
            )
        else:
            bctNNLOAB = np.zeros(3)
    else:
        bctAB = np.array(
            [-cctA - cctB, -(cr1A + cr1B) * f, -(cr2A + cr2B) * f**2, 0.0, 0.0, 0.0]
        )
        if bird.co.with_NNLO:
            ctilde, *_ = cnnloA
            bctNNLOAB = ctilde * np.array(
                [-(b1A**2) * f**4, -2 * b1A * f**5, -(f**6)]
            )
        else:
            bctNNLOAB = np.zeros(3)
    bloopAB = np.array(
        [
            1.0,
            1.0 / 2.0 * (b1A + b1B),
            1.0 / 2.0 * (b2A + b2B),
            1.0 / 2.0 * (b3A + b3B),
            1.0 / 2.0 * (b4A + b4B),
            b1A * b1B,
            1.0 / 2.0 * (b1A * b2B + b1B * b2A),
            1.0 / 2.0 * (b1A * b3B + b1B * b3A),
            1.0 / 2.0 * (b1A * b4B + b1B * b4A),
            b2A * b2B,
            1.0 / 2.0 * (b2A * b4B + b2B * b4A),
            b4A * b4B,
        ]
    )
    xfactor1 = 0.5 * (1.0 / ndA + 1.0 / ndB)
    xfactor2 = 0.5 * (1.0 / ndA / kmA**2 + 1.0 / ndB / kmB**2)
    bstAB = np.array([ce0 * xfactor1, cemono * xfactor2, cequad * xfactor2])

    No = bird.co.No
    Plin = np.einsum("b,lbx->lx", b11AB, bird.P11l[:No])
    Ploop = np.einsum("b,lbx->lx", bloopAB, bird.Ploopl[:No])
    Pct = np.einsum("b,lbx->lx", bctAB, bird.Pctl[:No])
    if bird.co.with_NNLO:
        assert bird.PctNNLOl is not None
        Pct += np.einsum("b,lbx->lx", bctNNLOAB, bird.PctNNLOl[:No])
    Pst = np.einsum("b,lbx->lx", bstAB, bird.Pstl[:No])
    return BirdComponent(Plin=Plin, Ploop=Ploop, Pct=Pct, Pst=Pst, Picc=bird.Picc[:No])


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

    def reduce_Plk(
        self, bird: BirdLike, params_values_dict: Mapping[str, float]
    ) -> BirdComponent:
        ...

    def reduce_Plk_gaussian_table(
        self,
        bird: BirdLike,
        params_values_dict: Mapping[str, float],
        requires: Container[str] | None = None,
    ) -> dict[str, ndarrayf]:
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

    def cnnloA(self) -> list[str]:
        return [self.prefix + p for p in ("cr4", "cr6")]

    def cnnloB(self) -> list[str]:
        raise NotImplementedError

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
            # NOTE: It's okay to include nnlo params here, since they will not be
            # included in the gaussian table.
            retval += self.cnnloA()
        return retval

    # impl
    def reduce_Plk(self, bird: BirdLike, params_values_dict: Mapping[str, float]):
        param_values = self.default()
        param_values.update(params_values_dict)
        bsA = [param_values[p] for p in self.bsA()]
        bsB = [param_values[p] for p in self.bsB()] or None
        es = [param_values[p] for p in self.es()]
        cnnloA = (
            [param_values[p] for p in self.cnnloA()]
            if bird.co.with_NNLO
            else (0.0, 0.0)
        )
        return reduce_Plk(bird, bsA, bsB, es, cnnloA)

    # impl
    def reduce_Plk_gaussian_table(
        self,
        bird: BirdLike,
        params_values_dict: Mapping[str, float],
        requires: Container[str] | None = None,
    ):
        return self.derivative_table(bird, params_values_dict, requires)

    def derivative_table(
        self,
        bird: BirdLike,
        params_values_dict: Mapping[str, float],
        requires: Container[str] | None = None,
    ):
        No = bird.co.No
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
        Ploopl, Pctl, PctNNLOl, Pstl = (
            _[:No] for _ in (bird.Ploopl, bird.Pctl, bird.PctNNLOl, bird.Pstl)
        )

        if requires is None:
            requires = EVERYTHING
        PG: dict[str, ndarrayf] = {}
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
            if bird.co.with_NNLO:
                if (p := self.prefix + "cr4") in requires:
                    PG[p] = 1 / 4 * b1A**2 / krA**4 * PctNNLOl[:, 0, :]
                if (p := self.prefix + "cr6") in requires:
                    PG[p] = 1 / 4 * b1A / krA**4 * PctNNLOl[:, 1, :]
        xfactor1 = 0.5 * (1.0 / ndA + 1.0 / ndB)
        xfactor2 = 0.5 * (1.0 / ndA / kmA**2 + 1.0 / ndB / kmB**2)
        if (p := self.prefix + "ce0") in requires:
            PG[p] = Pstl[:, 0, :] * xfactor1
        if (p := self.prefix + "cemono") in requires:
            PG[p] = Pstl[:, 1, :] * xfactor2
        if (p := self.prefix + "cequad") in requires:
            PG[p] = Pstl[:, 2, :] * xfactor2
        return PG


@dataclass(frozen=True)
class EastCoastBasis(EFTBasis):
    """

    Notes
    -----
    arxiv: 2106.12580, arxiv: 2208.05929
    """

    prefix: str = ""
    cross_prefix: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.cross_prefix:
            raise NotImplementedError("EastCoastBasis does not support cross yet")

    def default(self) -> dict[str, float]:
        return {p: 0.0 for p in self.gaussian_params()}

    def bsA(self) -> list[str]:
        names = ("b1", "b2", "bG2", "bGamma3", "c0", "c2", "c4")
        return [self.prefix + p for p in names]

    def es(self) -> list[str]:
        return [self.prefix + p for p in ("Pshot", "a0", "a2")]

    def cnnloA(self) -> list[str]:
        return [self.prefix + "ctilde"]

    def cnnloB(self) -> list[str]:
        raise NotImplementedError

    def is_cross(self) -> bool:
        return True if self.cross_prefix else False

    # override
    @classmethod
    def get_name(cls) -> str:
        return "eastcoast"

    # impl
    @classmethod
    def counterform(cls):
        return "eastcoast"

    # impl
    def non_gaussian_params(self) -> list[str]:
        names = ("b1", "b2", "bG2")
        return [self.prefix + p for p in names]

    # impl
    def gaussian_params(self) -> list[str]:
        names = ("bGamma3", "c0", "c2", "c4", "Pshot", "a0", "a2")
        retval = [self.prefix + p for p in names]
        # NOTE: It's okay to include nnlo params here, since they will not be
        # included in the gaussian table.
        retval += self.cnnloA()
        return retval

    # impl
    def reduce_Plk(self, bird: BirdLike, params_values_dict: Mapping[str, float]):
        param_values = self.default()
        param_values.update(params_values_dict)
        b1, b2, bG2, bGamma3, c0, c2, c4 = [param_values[p] for p in self.bsA()]
        f = bird.f
        bsA = [
            b1,
            b1 + 7 / 2 * bG2,
            b1 + 15 * bG2 + 6 * bGamma3,
            1 / 2 * b2 - 7 / 2 * bG2,
            c0 - f / 3 * c2 + 3 / 35 * f**2 * c4,
            c2 - 6 / 7 * f * c4,
            c4,
        ]
        bsB = None
        Pshot, a0, a2 = (param_values[p] for p in self.es())
        es = [Pshot, a0 + 1 / 3 * a2, 2 / 3 * a2]
        if bird.co.with_NNLO:
            cnnloA = [param_values[p] for p in self.cnnloA()] + [0.0]  # placeholder
        else:
            cnnloA = (0.0, 0.0)
        return reduce_Plk(bird, bsA, bsB, es, cnnloA=cnnloA)

    # impl
    def reduce_Plk_gaussian_table(
        self,
        bird: BirdLike,
        params_values_dict: Mapping[str, float],
        requires: Container[str] | None = None,
    ):
        No = bird.co.No
        f = bird.f
        kmA, krA, ndA, kmB, krB, ndB = (
            bird.co.kmA,
            bird.co.krA,
            bird.co.ndA,
            bird.co.kmB,
            bird.co.krB,
            bird.co.ndB,
        )
        b1A = b1B = params_values_dict[self.prefix + "b1"]
        Ploopl, Pctl, PctNNLOl, Pstl = (
            _[:No] for _ in (bird.Ploopl, bird.Pctl, bird.PctNNLOl, bird.Pstl)
        )

        if requires is None:
            requires = EVERYTHING
        PG: dict[str, ndarrayf] = {}
        if (p := self.prefix + "bGamma3") in requires:
            PG[p] = 6.0 * (Ploopl[:, 3, :] + b1A * Ploopl[:, 7, :])
        if (p := self.prefix + "c0") in requires:
            PG[p] = -2.0 * Pctl[:, 0, :]
        if (p := self.prefix + "c2") in requires:
            PG[p] = 2 / 3 * f * Pctl[:, 0, :] - 2.0 * f * Pctl[:, 1, :]
        if (p := self.prefix + "c4") in requires:
            PG[p] = (
                -6 / 35 * f**2 * Pctl[:, 0, :]
                + 12 / 7 * f**2 * Pctl[:, 1, :]
                - 2.0 * f**2 * Pctl[:, 2, :]
            )
        if bird.co.with_NNLO:
            if (p := self.prefix + "ctilde") in requires:
                PG[p] = (
                    -(b1A**2) * f**4 * PctNNLOl[:, 0, :]
                    - 2.0 * b1A * f**5 * PctNNLOl[:, 1, :]
                    - f**6 * PctNNLOl[:, 2, :]
                )
        xfactor1 = 0.5 * (1.0 / ndA + 1.0 / ndB)
        xfactor2 = 0.5 * (1.0 / ndA / kmA**2 + 1.0 / ndB / kmB**2)
        if (p := self.prefix + "Pshot") in requires:
            PG[p] = xfactor1 * Pstl[:, 0, :]
        if (p := self.prefix + "a0") in requires:
            PG[p] = xfactor2 * Pstl[:, 1, :]
        if (p := self.prefix + "a2") in requires:
            PG[p] = xfactor2 / 3 * (Pstl[:, 1, :] + 2.0 * Pstl[:, 2, :])
        return PG


def find_param_basis(name: str) -> Type[EFTBasis]:
    if name == "westcoast":
        return WestCoastBasis
    elif name == "eastcoast":
        return EastCoastBasis
    else:
        module_name, class_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
