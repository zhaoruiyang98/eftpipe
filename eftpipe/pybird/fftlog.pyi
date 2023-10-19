import numpy as np
from typing import Literal
from numpy.typing import NDArray

type ExtrapT = Literal["extrap", "padding"]
type NDArrayFloat = NDArray[np.float64]
type NDArrayComplex = NDArray[np.complex128]

def one(x) -> Literal[1]: ...
def CoefWindow(
    N: int, window: float = 1, left: bool = True, right: bool = True
) -> NDArrayFloat: ...

class FFTLog:
    xmin: float
    xmax: float
    bias: float
    dx: float
    x: NDArrayFloat
    Pow: NDArrayComplex

    def __init__(self, Nmax: int, xmin: float, xmax: float, bias: float) -> None: ...
    def setx(self) -> None: ...
    def setPow(self) -> None: ...
    def setCoefFactor(self) -> None: ...
    def Coef(
        self,
        xin: NDArrayFloat,
        f,
        extrap: ExtrapT | tuple[ExtrapT, ExtrapT] = "extrap",
        window: float | None = 1,
        log_interp: bool = False,
        kernel=one,
    ) -> NDArrayComplex: ...
    def sumCoefxPow(self, xin, f, x, window: float | None = 1): ...