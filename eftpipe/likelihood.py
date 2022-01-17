# global
import sys
from cobaya.likelihood import Likelihood
from cobaya.theory import Provider
from pathlib import Path
from typing import (
    List,
    Dict,
    Any,
    overload,
    Type,
)
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
# local
from eftpipe.tools import update_path_in_dict
from eftpipe.parser import (
    SingleTracerParser,
    TwoTracerParser,
    TwoTracerCrossParser,
)
from eftpipe.typing import (
    GaussianData,
    VectorTheory,
)


class EFTLike(Likelihood):
    label: str
    nsampled: int
    extra_args: Dict[str, Any]
    # intermediate products
    input_params: Dict[str, Any]
    data_obj: GaussianData
    theory_obj: VectorTheory

    def initialize(self) -> None:
        base_path = self.extra_args.get('base', None)
        if base_path is not None:
            update_path_in_dict(self.extra_args, Path(str(base_path)))
        parser = select_parser(self.extra_args['mode'])(
            self.extra_args, logfunc=self.mpi_info
        )
        data_obj = parser.create_gaussian_data()
        theory_obj = parser.create_vector_theory()
        self.data_obj = data_obj
        self.theory_obj = theory_obj
        if self.nsampled is None:
            raise ValueError(
                f"please specify the number of sampled parameters")
        if self.label is None:
            raise ValueError(f"please specify the label of likelihood")
        self.label = str(self.label)

    def initialize_with_provider(self, provider: 'Provider'):
        super().initialize_with_provider(provider)
        self.theory_obj.set_provider(self.provider)

    def get_requirements(self) -> Dict[str, Any]:
        return self.theory_obj.required_params()

    def calculate(self, state, want_derived=True, **params_values_dict):
        theory = self.theory_vector(**params_values_dict)
        res = theory - self.data_obj.data_vector
        chi2 = res @ self.data_obj.invcov @ res

        if want_derived:
            state['derived'] = {
                self.label + 'reduced_chi2':
                chi2 / (self.data_obj.ndata - self.nsampled)
            }
        state['logp'] = -0.5 * chi2

    def theory_vector(self, **params_values_dict):
        return self.theory_obj.theory_vector(params_values_dict)

    def get_can_provide_params(self) -> List[str]:
        return [self.label + 'reduced_chi2']


@overload
def select_parser(mode: Literal['single']) -> Type[SingleTracerParser]: ...
@overload
def select_parser(mode: Literal['two']) -> Type[TwoTracerParser]: ...
@overload
def select_parser(mode: Literal['all']) -> Type[TwoTracerCrossParser]: ...


def select_parser(mode: str):
    if mode == 'single':
        return SingleTracerParser
    elif mode == 'two':
        return TwoTracerParser
    elif mode == 'all':
        return TwoTracerCrossParser
    else:
        raise ValueError(f"unexpected mode {mode}")
