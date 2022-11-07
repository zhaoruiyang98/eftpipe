import cobaya
import numpy as np
from pathlib import Path
from eftpipe.tools import PathContext
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture


def unpack_PG_table(prefix: str, product):
    ls, kgrid, PG_table = product
    out = {f"{prefix}_{k}": v for k, v in PG_table.items()}
    out[f"{prefix}_ls"] = ls
    out[f"{prefix}_kgrid"] = kgrid
    return out


def test_ELG_NGC_reg(yamlroot: Path, ndarrays_regression: NDArraysRegressionFixture):
    info = yamlroot / "mock_eBOSS_ELG_NGC_theory.yaml"
    with PathContext("cobaya"):
        model = cobaya.get_model(info)
        kout = np.arange(25) * 0.01 + 0.005
        model.add_requirements(
            {
                "nonlinear_Plk_interpolator": {
                    "ELG_NGC": {"ls": [0, 2, 4], "chained": [False, True]}
                },
                "nonlinear_Plk_grid": {
                    "ELG_NGC": {
                        "ls": [0, 2, 4],
                        "chained": [False, True],
                        "binned": [False, True],
                        "binning": {"kout": kout},
                    }
                },
                "nonlinear_Plk_gaussian_grid": {
                    "ELG_NGC": {
                        "ls": [0, 2, 4],
                        "chained": [False, True],
                        "binned": [False, True],
                        "binning": {"kout": kout},
                    }
                },
                "ELG_NGC_fsigma8_z": None,
                "ELG_NGC_alperp": None,
                "ELG_NGC_alpara": None,
            }
        )
    sampled_dict = dict(
        omegach2=1.119507559e-01,
        H0=6.770409490e1,
        logA=2.815516798e00,
        ELG_NGC_b1=1.589354637e00,
        ELG_NGC_c2=1.266786101e00,
        ELG_NGC_b3=1.675518719e-01,
        ELG_NGC_cct=-7.109794839e-01,
        ELG_NGC_cr1=-2.635512354e00,
        ELG_NGC_cr2=-3.667395306e-01,
        ELG_NGC_ce0=1.634998078e-02,
        ELG_NGC_cequad=-4.125790673e-02,
    )
    model.logpost(sampled_dict)
    provider = model.provider

    # fmt: off
    reg_dct = {
        "plk": provider.get_nonlinear_Plk_interpolator("ELG_NGC")([0, 2, 4], kout),
        "plk_chained": provider.get_nonlinear_Plk_interpolator("ELG_NGC", chained=True)(
            [0, 2], kout
        ),
        "plk_grid": provider.get_nonlinear_Plk_grid("ELG_NGC")[-1],
        "plk_grid_chained": provider.get_nonlinear_Plk_grid("ELG_NGC", chained=True)[-1],
        "plk_grid_binned": provider.get_nonlinear_Plk_grid("ELG_NGC", chained=False, binned=True)[-1],
        "ELG_NGC_fsigma8_z": provider.get_param("ELG_NGC_fsigma8_z"),
        "ELG_NGC_alperp": provider.get_param("ELG_NGC_alperp"),
        "ELG_NGC_alpara": provider.get_param("ELG_NGC_alpara"),
        **unpack_PG_table("plk_grid_gaussian", provider.get_nonlinear_Plk_gaussian_grid("ELG_NGC", chained=True)),
        **unpack_PG_table("plk_grid_gaussian_chained", provider.get_nonlinear_Plk_gaussian_grid("ELG_NGC", chained=True)),
        **unpack_PG_table("plk_grid_gaussian_binned", provider.get_nonlinear_Plk_gaussian_grid("ELG_NGC", chained=False, binned=True)),
    }
    # fmt: on
    ndarrays_regression.check(reg_dct, default_tolerance={"atol": 0, "rtol": 1e-8})
