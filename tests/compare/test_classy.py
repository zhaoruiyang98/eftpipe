import numpy as np
import cobaya
import pytest
from pathlib import Path
from cobaya.yaml import yaml_load_file
from eftpipe.interface import CobayaClassyProvider


def test_compare_classynu_and_classy(
    compare_ndarrays, rtol: float, atol: float, yamlroot: Path
):
    classy_model = cobaya.get_model(yamlroot / 'classy_planck18.yaml')
    classynu_model = cobaya.get_model(yamlroot / 'classynu_planck18.yaml')

    z = 0.5
    requires = {
        "Pk_interpolator": {
            "nonlinear": False,
            "z": [z],
            "k_max": 5,
        },
        "Hubble": {'z': [0., z]},
        "angular_diameter_distance": {'z': [z]},
        "rdrag": None,
        'sigma8': None,
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

    classy_provider = CobayaClassyProvider(classy_model.provider, z)
    classynu_provider = CobayaClassyProvider(classynu_model.provider, z)

    dcts = []
    for provider in (classy_provider, classynu_provider):
        kh = np.logspace(-4, 0, 200)
        dct = {
            'rdrag': provider.get_rdrag(),
            'DA': provider.get_DA(z),
            'H': provider.get_H(z),
            'f': provider.get_f(z),
            'sigma8': provider.provider.get_param('sigma8'),
            'pkh': provider.interp_pkh(kh),
        }
        dcts.append(dct)
    ref, data = dcts
    compare_ndarrays(
        ref, data,
        default_tolerance={'atol': atol, 'rtol': rtol},
    )


@pytest.mark.fcompare
@pytest.mark.parametrize('mnu', [0.06, 0.08, 0.10])
def test_compare_hierarchy_normal_and_degenerate(
    compare_ndarrays, rtol: float, atol: float, yamlroot: Path, mnu: float,
):
    """
    For mnu=0.1, I have verified classy and camb gives quite similar results
    """
    info = yaml_load_file(str(yamlroot / 'classynu_planck18.yaml'))
    # info["theory"]["classynu"]["extra_args"]["ncdm_fluid_approximation"] = 3
    info["params"]["mnu"]["value"] = mnu
    # degenerate
    # uncomment the following line to study the mass split effect on the linear power spectrum
    # info["theory"]["classynu"]["extra_args"] = {"N_ncdm": 3, "N_ur": 0.00641}
    info["theory"]["classynu"]["extra_args"]["neutrino_hierarchy"] = "degenerate"
    degenerate_model = cobaya.get_model(info) # type: ignore
    # normal
    info["theory"]["classynu"]["extra_args"]["neutrino_hierarchy"] = "normal"
    normal_model = cobaya.get_model(info) # type: ignore

    z = 0.5
    requires = {
        "Pk_interpolator": {
            "nonlinear": False,
            "z": [z],
            "k_max": 5,
        },
        "Hubble": {'z': [0., z]},
        "angular_diameter_distance": {'z': [z]},
        "rdrag": None,
        'sigma8': None,
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

    degenerate_provider = CobayaClassyProvider(degenerate_model.provider, z)
    normal_provider = CobayaClassyProvider(normal_model.provider, z)

    dcts = []
    for provider in (degenerate_provider, normal_provider):
        kh = np.logspace(-4, 0, 200)
        dct = {
            'rdrag': provider.get_rdrag(),
            'DA': provider.get_DA(z),
            'H': provider.get_H(z),
            'f': provider.get_f(z),
            'sigma8': provider.provider.get_param('sigma8'),
            'pkh': provider.interp_pkh(kh),
        }
        dcts.append(dct)
    ref, data = dcts
    compare_ndarrays(
        ref, data,
        default_tolerance={'atol': atol, 'rtol': rtol},
    )
