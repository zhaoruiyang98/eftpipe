import cobaya
import numpy as np
from pathlib import Path
from eftpipe.tools import PathContext
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture


def test_test_ELG_NGC_likelihood_reg(
    yamlroot: Path, ndarrays_regression: NDArraysRegressionFixture
):
    info = yamlroot / "mock_eBOSS_ELG_NGC_like.yaml"
    with PathContext("cobaya"):
        model = cobaya.get_model(info)
    sampled_dict = dict(
        omegach2=1.118146487e-01,
        H0=6.738083803e1,
        logA=3.10113145e00,
        ELG_NGC_b1=1.264557666e00,
        ELG_NGC_c2=4.471452265e-01,
    )
    logpost = model.logpost(sampled_dict)
    likelihood = model.likelihood["ELG_NGC"]
    regdict = dict(
        logpost=logpost,
        data_vector=likelihood.data_vector,
        invcov=likelihood.invcov,
        PNG=likelihood.PNG(),
        PG=likelihood.PG(),
    )
    ndarrays_regression.check(regdict, default_tolerance={"atol": 0, "rtol": 1e-8})
