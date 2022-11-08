import numpy as np
import cobaya
import pytest
from pathlib import Path
from cobaya.yaml import yaml_load_file
from eftpipe.interface import CobayaClassyInterface


def test_compare_classynu_and_classy(
    compare_ndarrays, rtol: float, atol: float, yamlroot: Path
):
    classy_model = cobaya.get_model(yamlroot / "classy_planck18.yaml")
    classynu_model = cobaya.get_model(yamlroot / "classynu_planck18.yaml")

    z = 0.5
    zs = [z + 0.1 * i for i in range(5)]
    requires = {
        "Pk_interpolator": {
            "nonlinear": False,
            "z": zs,
            "k_max": 5,
        },
        "Hubble": {"z": [0.0, *zs]},
        "angular_diameter_distance": {"z": zs},
        "rdrag": None,
        "sigma8": None,
    }
    classy_model.add_requirements(requires)
    classynu_model.add_requirements(requires)

    point = {
        "logA": 3.05,
        "omega_b": 0.02214,
        "omega_cdm": 0.12,
    }

    classy_model.logpost(point)
    classynu_model.logpost(point)

    classy_provider = CobayaClassyInterface()
    classy_provider.initialize(z)
    classy_provider.initialize_with_provider(classy_model.provider)
    classynu_provider = CobayaClassyInterface()
    classynu_provider.initialize(z)
    classynu_provider.initialize_with_provider(classynu_model.provider)

    dcts = []
    for provider in (classy_provider, classynu_provider):
        kh = np.logspace(-4, 0, 200)
        dct = {
            "rdrag": provider.rdrag(),
            "DA": provider.DA(),
            "H": provider.H(),
            "f": provider.f(),
            "sigma8": provider.provider.get_param("sigma8"),
            "pkh": provider.Pkh(kh),
        }
        dcts.append(dct)
    ref, data = dcts
    compare_ndarrays(
        ref,
        data,
        default_tolerance={"atol": atol, "rtol": rtol},
    )


@pytest.mark.fcompare
@pytest.mark.parametrize("mnu", [0.06, 0.08, 0.10])
def test_compare_hierarchy_normal_and_degenerate(
    compare_ndarrays,
    rtol: float,
    atol: float,
    yamlroot: Path,
    mnu: float,
):
    info = yaml_load_file(str(yamlroot / "classynu_planck18.yaml"))
    # info["theory"]["eftpipe.classynu"]["extra_args"]["ncdm_fluid_approximation"] = 3
    info["params"]["mnu"]["value"] = mnu
    # degenerate
    # uncomment the following line to study the mass split effect on the linear power spectrum
    # info["theory"]["eftpipe.classynu"]["extra_args"] = {"N_ncdm": 3, "N_ur": 0.00641}
    info["theory"]["eftpipe.classynu"]["extra_args"][
        "neutrino_hierarchy"
    ] = "degenerate"
    degenerate_model = cobaya.get_model(info)  # type: ignore
    # normal
    info["theory"]["eftpipe.classynu"]["extra_args"]["neutrino_hierarchy"] = "normal"
    normal_model = cobaya.get_model(info)  # type: ignore

    z = 0.5
    zs = [z + 0.1 * i for i in range(5)]
    requires = {
        "Pk_interpolator": {
            "nonlinear": False,
            "z": zs,
            "k_max": 5,
        },
        "Hubble": {"z": [0.0, *zs]},
        "angular_diameter_distance": {"z": zs},
        "rdrag": None,
        "sigma8": None,
    }

    degenerate_model.add_requirements(requires)
    normal_model.add_requirements(requires)

    point = {
        "logA": 3.05,
        "omega_b": 0.02214,
        "omega_cdm": 0.12,
    }

    degenerate_model.logpost(point)
    normal_model.logpost(point)

    degenerate_provider = CobayaClassyInterface()
    degenerate_provider.initialize(z)
    degenerate_provider.initialize_with_provider(degenerate_model.provider)
    normal_provider = CobayaClassyInterface()
    normal_provider.initialize(z)
    normal_provider.initialize_with_provider(normal_model.provider)

    dcts = []
    for provider in (degenerate_provider, normal_provider):
        kh = np.logspace(-4, 0, 200)
        dct = {
            "rdrag": provider.rdrag(),
            "DA": provider.DA(),
            "H": provider.H(),
            "f": provider.f(),
            "sigma8": provider.provider.get_param("sigma8"),
            "pkh": provider.Pkh(kh),
        }
        dcts.append(dct)
    ref, data = dcts
    compare_ndarrays(
        ref,
        data,
        default_tolerance={"atol": atol, "rtol": rtol},
    )
