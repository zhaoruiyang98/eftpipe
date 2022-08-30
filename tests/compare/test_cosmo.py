import numpy as np
import cobaya
import pytest
from pathlib import Path
from eftpipe.interface import CobayaClassyInterface, CobayaCambInterface


@pytest.mark.fcompare
@pytest.mark.parametrize("use_cb", (False, True), ids=["no_cb", "cb"])
def test_compare_camb_and_classy(
    compare_ndarrays, rtol: float, atol: float, yamlroot: Path, use_cb: bool,
):
    camb_model = cobaya.get_model(str(yamlroot / "camb_planck18.yaml"))
    classy_model = cobaya.get_model(str(yamlroot / "classy_planck18.yaml"))

    z = 0.5
    zs = [z + 0.1 * i for i in range(-2, 3)]
    vars_pairs = [2 * ["delta_nonu"]] if use_cb else [2 * ["delta_tot"]]
    requires = {
        "Pk_interpolator": {
            "nonlinear": False,
            "z": zs,
            "k_max": 5,
            "vars_pairs": vars_pairs,
        },
        "Hubble": {"z": [0.0, *zs]},
        "angular_diameter_distance": {"z": zs},
        "rdrag": None,
        "sigma8": None,
        "fsigma8": {"z": zs},
    }
    classy_model.add_requirements(requires)
    requires["sigma8_z"] = {"z": zs}
    camb_model.add_requirements(requires)

    classy_point = {
        "logA": 3.05,
        "omega_b": 0.02214,
        "omega_cdm": 0.12,
    }
    camb_point = {
        "logA": 3.05,
        "ombh2": 0.02214,
        "omch2": 0.12,
    }

    classy_model.logpost(classy_point)
    camb_model.logpost(camb_point)

    classy_provider = CobayaClassyInterface(classy_model.provider, z, use_cb=use_cb)
    camb_provider = CobayaCambInterface(camb_model.provider, z, use_cb=use_cb)

    dcts = []
    for provider in (camb_provider, classy_provider):
        kh = np.logspace(-5, 0, 200)
        dct = {
            "rdrag": provider.rdrag,
            "DA": provider.DA,
            "H": provider.H,
            "f": provider.f,
            "sigma8": provider.provider.get_param("sigma8"),
            "pkh": provider.Pkh(kh),
        }
        dcts.append(dct)
    ref, data = dcts
    compare_ndarrays(
        ref, data, default_tolerance={"atol": atol, "rtol": rtol},
    )

