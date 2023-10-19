import numpy as np
from cobaya.log import HasLogger
from collections.abc import Callable
from typing import Literal, Protocol
from numpy.typing import ArrayLike, NDArray

from .fftlog import FFTLog

type NDArrayFloat = NDArray[np.float64]
type NDArrayComplex = NDArray[np.complex128]

def cH(Om, a): ...
def DgN(Om, a): ...
def fN(Om, z): ...
def Hubble(Om, z): ...
def DAfunc(Om, z): ...
def W2D(x): ...
def Hllp(l: int, lp: int, x): ...
def fllp_IR(l, lp, k, q, Dfc): ...
def fllp_UV(l, lp, k, q, Dfc): ...

mu: dict[int, dict[int, float]] = ...
M13b: dict[int, Callable] = ...

def M13a(n1): ...

M22b: dict[int, Callable] = ...

def M22a(n1, n2): ...
def MPC(l, pn): ...

Q000: dict[int, Callable[[float], float]] = ...
Q002: dict[int, Callable[[float], float]] = ...
Q020: dict[int, Callable[[float], float]] = ...
Q022: dict[int, Callable[[float], float]] = ...
Q100: dict[int, Callable[[float], float]] = ...
Q102: dict[int, Callable[[float], float]] = ...
Q120: dict[int, Callable[[float], float]] = ...
Q122: dict[int, Callable[[float], float]] = ...
Qa: dict[int, dict[int, dict[int, dict[int, Callable[[float], float]]]]] = ...

def get_kbird(kmax: float = 0.30) -> NDArrayFloat: ...

sbird: NDArrayFloat = ...

class Common:
    optiresum: bool
    kmA: float
    krA: float
    ndA: float
    kmB: float
    krB: float
    ndB: float
    counterform: Literal["westcoast", "eastcoast"]
    with_NNLO: bool
    IRcutoff: Literal["all", "resum", "loop"] | bool
    kIR: float
    Nl: int
    No: int
    N11: int
    Nct: int
    NctNNLO: int
    N22: int
    N13: int
    Nloop: int
    k: NDArrayFloat
    Nk: int
    s: NDArrayFloat
    Ns: int
    kr: NDArrayFloat
    Nkr: int
    Nklow: int

    l11: NDArrayFloat
    lct: NDArrayFloat
    lctNNLO: NDArrayFloat
    l22: NDArrayFloat
    l13: NDArrayFloat

    def __init__(
        self,
        Nl: int | None = None,
        No: int | None = None,
        kmax: float = 0.3,
        optiresum: bool = False,
        kmA: float = 0.7,
        krA: float = 0.25,
        ndA: float = 3e-4,
        kmB: float | None = None,
        krB: float | None = None,
        ndB: float | None = None,
        counterform: Literal["westcoast", "eastcoast"] = "westcoast",
        with_NNLO: bool = False,
        kIR: float | None = None,
        IRcutoff: Literal["all", "resum", "loop"] | bool = False,
    ) -> None: ...

common: Common = ...

class BirdLike(Protocol):
    co: Common
    f: float
    P11l: NDArrayFloat
    Ploopl: NDArrayFloat
    Pctl: NDArrayFloat
    Pstl: NDArrayFloat
    Picc: NDArrayFloat
    PctNNLOl: NDArrayFloat


class BirdSnapshot(BirdLike):
    co: Common
    f: float
    P11l: NDArrayFloat
    Ploopl: NDArrayFloat
    Pctl: NDArrayFloat
    Pstl: NDArrayFloat
    Picc: NDArrayFloat
    PctNNLOl: NDArrayFloat
    def __init__(self, bird: Bird) -> None: ...

class Bird:
    co: Common
    f: float
    DA: float
    H: float
    z: float
    rdrag: float
    h: float

    kin: NDArrayFloat
    Pin: NDArrayFloat
    Plin: Callable[[ArrayLike], NDArrayFloat]
    P11: NDArrayFloat
    P22: NDArrayFloat
    P13: NDArrayFloat
    Ploopl: NDArrayFloat
    Cloopl: NDArrayFloat
    P11l: NDArrayFloat
    Pctl: NDArrayFloat
    PctNNLOl: NDArrayFloat
    P22l: NDArrayFloat
    P13l: NDArrayFloat
    Pstl: NDArrayFloat
    C11: NDArrayFloat
    C22: NDArrayFloat
    C13: NDArrayFloat
    Cct: NDArrayFloat
    CctNNLO: NDArrayFloat
    b11: NDArrayFloat
    b13: NDArrayFloat
    b22: NDArrayFloat
    bct: NDArrayFloat
    Ps: NDArrayFloat
    Cf: NDArrayFloat
    Picc: NDArrayFloat

    snapshots: dict[str, BirdSnapshot]
    def __init__(
        self,
        kin,
        Plin,
        f,
        DA=None,
        H=None,
        z=None,
        co=common,
        rdrag: float | None = None,
        h: float | None = None,
    ) -> None: ...
    def create_snapshot(self, name: str): ...
    def setPsCfl(self): ...
    def reducePsCfl(self): ...
    def setPstl(self): ...
    def subtractShotNoise(self): ...

class NonLinear(HasLogger):
    co: Common
    fftsettings: dict
    fft: FFTLog
    M22: NDArrayComplex
    M13: NDArrayComplex
    Mcf11: NDArrayComplex
    Ml: NDArrayComplex
    Mcf22: NDArrayComplex
    Mcf13: NDArrayComplex
    Mcfct: NDArrayComplex
    McfctNNLO: NDArrayComplex
    kPow: NDArrayComplex
    sPow: NDArrayComplex

    def __init__(
        self,
        load: bool = True,
        save: bool = True,
        path: str = "./",
        NFFT: int = 256,
        co: Common = common,
        name: str = "pybird.nonlinear",
    ) -> None: ...
    def setM22(self) -> None: ...
    def setM13(self) -> None: ...
    def setMcf11(self) -> None: ...
    def setMl(self) -> None: ...
    def setMcf22(self) -> None: ...
    def setMcf13(self) -> None: ...
    def setMcfct(self) -> None: ...
    def setMcfctNNLO(self) -> None: ...
    def setkPow(self) -> None: ...
    def setsPow(self) -> None: ...
    def CoefkPow(self, Coef): ...
    def CoefsPow(self, Coef): ...
    def makeP22(self, CoefkPow, bird): ...
    def makeP13(self, CoefkPow, bird): ...
    def makeC11(self, CoefsPow, bird): ...
    def makeCct(self, CoefsPow, bird): ...
    def makeCctNNLO(self, CoefsPow, bird): ...
    def makeC22(self, CoefsPow, bird): ...
    def makeC13(self, CoefsPow, bird): ...
    def Coef(self, bird, window: float | None = None, IRcut: bool = False): ...
    def PsCf(self, bird, window: float | None = 0.2): ...

class Resum(HasLogger):
    co: Common
    LambdaIR: float
    slow: float
    sHigh: float
    idlow: int
    idhigh: int
    sbao: NDArrayFloat
    snobao: NDArrayFloat
    sr: NDArrayFloat

    NIR: int
    Na: int
    Nn: int
    k2p: NDArrayFloat

    alllpr: list[list[int]]
    Q: NDArrayFloat
    IR11: NDArrayFloat
    IRct: NDArrayFloat
    IRctNNLO: NDArrayFloat
    IRloop: NDArrayFloat
    
    fftsettings: dict
    fft: FFTLog

    Xfftsettings: dict
    Xfft: FFTLog

    XsPow: NDArrayComplex
    kPow: NDArrayComplex
    XM: NDArrayComplex
    M: NDArrayComplex

    snapshot: bool

    def __init__(
        self,
        LambdaIR: float = 0.2,
        NFFT: int = 192,
        co: Common = common,
        name: str = "pybird.IRresum",
        snapshot: bool = False,
    ) -> None: ...
    def setXsPow(self) -> None: ...
    def setkPow(self) -> None: ...
    def setXM(self) -> None: ...
    def IRFilters(
        self,
        bird,
        soffset=1.0,
        LambdaIR=None,
        RescaleIR=1.0,
        window=None
    ) -> tuple[NDArrayFloat, NDArrayFloat]: ...
    def setM(self) -> None: ...
    def IRnWithCoef(self, Coef): ...
    def makeQ(self, f: float) -> None: ...
    def extractBAO(self, cf): ...
    def setXpYp(self, bird): ...
    def precomputedCoef(self, XpYp, Carray, window=None) -> NDArrayComplex: ...
    def Ps(self, bird: Bird, window=None): ...

class APeffect(HasLogger):
    co: Common
    APst: bool
    DA: float
    H: float
    rdrag_AP: float
    h_AP: float
    nbinsmu: int
    muacc: NDArrayFloat
    kgrid: NDArrayFloat
    mugrid: NDArrayFloat
    Nlmax: int
    arrayLegendremugrid: NDArrayFloat
    snapshot: bool

    def __init__(
        self,
        Om_AP: float | None = None,
        z_AP: float | None = None,
        DA: float | None = None,
        H: float | None = None,
        rdrag_AP: float | None = None,
        h_AP: float | None = None,
        nbinsmu: int = 200,
        accboost: int = 1,
        Nlmax: int | None = None,
        APst: bool = False,
        co: Common = common,
        name: str = "pybird.apeffect",
        snapshot: bool = False,
    ): ...

    def get_AP_param(self, bird) -> tuple[float, float]: ...
    def get_alperp_alpara(self, bird) -> tuple[float, float]: ...
    def integrAP(self, Pk, kp, arrayLegendremup) -> NDArrayFloat: ...
    def AP(self, bird: Bird, q: tuple[float, float] | None = None) -> None: ...
    def logstate(self): ...


class FiberCollision(HasLogger):
    ktrust: float
    fiberst: bool
    fs: float
    Dfc: float
    co: Common
    snapshot: bool

    def __init__(
        self,
        fs: float,
        Dfc: float,
        ktrust: float = 0.25,
        fiberst: bool = False,
        co: Common = common,
        name: str = "pybird.fiber",
        snapshot: bool = False,
    ) -> None: ...
    def dPuncorr(self, kout, fs=0.6, Dfc=0.43 / 0.6777) -> NDArrayFloat: ...
    def dPcorr(self, kout, kPS, PS, ktrust=0.25, fs=0.6, Dfc=0.43 / 0.6777) -> NDArrayFloat: ...
    def fibcolWindow(self, bird: Bird) -> None: ...