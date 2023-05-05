from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING
from .pybird.pybird import BirdLike

if TYPE_CHECKING:
    from .pybird.pybird import Common
    from .etyping import ndarrayf


class BirdTransformer(Protocol):
    def transform(self, birdlike: BirdLike) -> BirdLike:
        ...


@dataclass
class PlainBird(BirdLike):
    f: float
    co: Common
    P11l: ndarrayf
    Ploopl: ndarrayf
    Pctl: ndarrayf
    Pstl: ndarrayf
    Picc: ndarrayf
    PctNNLOl: ndarrayf


class BirdCopier(BirdTransformer):
    def transform(self, birdlike: BirdLike):
        return PlainBird(
            f=birdlike.f,
            co=birdlike.co,
            P11l=birdlike.P11l.copy(),
            Ploopl=birdlike.Ploopl.copy(),
            Pctl=birdlike.Pctl.copy(),
            Pstl=birdlike.Pstl.copy(),
            Picc=birdlike.Picc.copy(),
            PctNNLOl=birdlike.PctNNLOl.copy(),
        )
