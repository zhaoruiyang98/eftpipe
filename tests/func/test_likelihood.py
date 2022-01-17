import cobaya
import os
from pathlib import Path

def test_EFTLike():
    current = Path('.').cwd()
    os.chdir(current / 'cobaya')
    
    model = cobaya.get_model(
        'yamls/mock_LRG_ELG_x_NGC_km0p15_fix_proposal.yaml',
        debug=True
    )
    point = model.prior.sample(ignore_external=True)[0]
    params_name = model.parameterization.sampled_params().keys()
    point_dct = {key: value for key, value in zip(params_name, point)}
    out = model.logpost(point_dct)

    os.chdir(current)