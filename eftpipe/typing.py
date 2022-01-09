import sys
from numpy import ndarray as NDArray
from pathlib import Path
from typing import (
    Union,
    List,
    Optional,
    Dict,
    Any,
    Callable,
)
if sys.version_info >= (3, 8):
    from typing import Protocol, TypedDict
else:
    from typing_extensions import Protocol, TypedDict

Location = Union[str, Path]


class GaussianData(Protocol):
    ndata: int
    data_vector: NDArray
    invcov: NDArray


class VectorTheory(Protocol):
    def required_params(self) -> Dict[str, Any]: ...
    def theory_vector(self, all_params_dict: Dict[str, Any]) -> NDArray: ...
    def set_provider(self, provider) -> None: ...


class CroppedPklDataDict(TypedDict):
    kmin: float
    kmax: float
    ls: Union[int, List[int]]
    pkl_path: str


class PartialCroppedPklDataDict(TypedDict, total=False):
    kmin: float
    kmax: float
    ls: Union[int, List[int]]
    pkl_path: str


class FullShapeDataDict(TypedDict):
    pklinfo: Union[CroppedPklDataDict, List[CroppedPklDataDict]]
    cov_path: str
    common: Optional[PartialCroppedPklDataDict]
    Nreal: Optional[int]


class ProjectionConfig(TypedDict, total=False):
    Om_AP: float
    z_AP: float
    kdata: NDArray
    windows_fourier_path: Path
    windows_configspace_path: Path
    ktrust: float
    fs: float
    Dfc: float
    integral_constraint_path: Path
    shotnoise: float


class BoltzmannProvider(Protocol):
    def interp_pkh(self, kh: NDArray) -> NDArray: ...
    def get_h0(self) -> float: ...
    def get_Hubble(self, z: float) -> float: ...
    def get_angular_diameter_distance(self, z: float) -> float: ...
    def get_fsigma8(self, z: float) -> float: ...
    def get_sigma8_z(self, z: float) -> float: ...
    def cosmo_updated(self) -> bool: ...


class EFTTheoryConfig(TypedDict, total=False):
    z: bool
    cache_dir: Path
    optiresum: bool
    cross: bool
    projection_config: ProjectionConfig
    bolzmann_provider: BoltzmannProvider
    print_info: Callable[[str], None]
