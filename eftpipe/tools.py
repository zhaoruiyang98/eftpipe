from typing import Any, List, TYPE_CHECKING


class SampledParamsMixin:

    @property
    def nsampled(self) -> int:
        return len(self.sampled_params)

    @property
    def sampled_params(self) -> List[str]:
        input_params = set(self.input_params)
        sampled_params = set(
            self.provider.model.parameterization.sampled_params().keys()
        )
        return list(input_params.intersection(sampled_params))

    if TYPE_CHECKING:
        def __getattr__(self, __name: str) -> Any: ...