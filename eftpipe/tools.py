from pathlib import Path
from typing import Any, List, TYPE_CHECKING
from eftpipe.typing import SimpleYaml

# This Mixin is not reliable, don't use it
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


def update_path_in_dict(d: SimpleYaml, base: Path) -> None:
    if isinstance(d, dict):
        for key, value in d.items():
            if not isinstance(value, (dict, list)):
                if 'path' in key:
                    d[key] = str(base / str(d[key]))
            else:
                update_path_in_dict(value, base)
    if isinstance(d, list):
        for item in d:
            update_path_in_dict(item, base)
