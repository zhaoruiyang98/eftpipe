import cobaya
from pathlib import Path
from eftpipe.tools import PathContext

def test_EFTLike():
    current = Path('.').cwd()
    yaml_file = current / "tests" / "yamls" / "mock_LRG_ELG_x_NGC_km0p15_fix_proposal.yaml"

    with PathContext("cobaya"):
        model = cobaya.get_model(yaml_file, debug=True)
    point = model.prior.sample(ignore_external=True)[0]
    params_name = model.parameterization.sampled_params().keys()
    point_dct = {key: value for key, value in zip(params_name, point)}
    out = model.logpost(point_dct)