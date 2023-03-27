from __future__ import annotations
from typing import Protocol
from .pybird.pybird import BirdLike


class BirdTransformer(BirdLike, Protocol):
    def transform(self, birdlike: BirdLike) -> BirdTransformer:
        ...
