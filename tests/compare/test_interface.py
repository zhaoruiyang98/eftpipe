import numpy as np
import cobaya
import pytest
from pathlib import Path
from eftpipe.interface import CambProvider
from eftpipe.interface import CobayaCambProvider
from eftpipe.interface import CobayaClassyProvider


def two_providers():
    info = r"""
    theory:
        camb:
            extra_args:
                nnu: 3.046
    likelihood:
        one:
    params:
        ombh2:
            prior:
                min: 0.01
                max: 0.3
            ref: 0.02214
            latex: \Omega_\mathrm{b} h^2
        omch2:
            prior:
                min: 0.03
                max: 0.7
            proposal: 0.003
            ref: 0.1189
            latex: \Omega_\mathrm{c} h^2
        H0:
            prior:
                min: 40.0
                max: 100.0
            proposal: 0.3
            ref: 67.77
            latex: H_0
        logA:
            prior:
                min: 0.1
                max: 10
            proposal: 0.1
            ref: 3.0
            latex: \log(10^{10} A_\mathrm{s})
            drop: true
        As:
            value: 'lambda logA: 1e-10*np.exp(logA)'
            latex: A_\mathrm{s}
        ns:
            prior:
                min: 0.8
                max: 1.1
            ref: 0.9611
            latex: n_\mathrm{s}
        mnu:
            prior:
                min: 0.0
                max: 1.0
            ref: 0.0
            latex: \sum m_\nu
        tau:
            prior:
                min: 0.01
                max: 0.1
            ref: 0.0543
            latex: \tau_\mathrm{reio}
    """
    params_dict = {
        'ombh2': 0.02214,
        'omch2': 1.111943844e-01,
        'H0': 6.630797882e+01,
        'logA': 3.129647579e+00,
        'ns': 0.9611,
        'mnu': 0.0,
        'tau': 0.0543,
    }
    z = 0.7
    model = cobaya.get_model(info)
    model.add_requirements({
        'Pk_grid': {
            'nonlinear': False,
            'z': [z],
            'k_max': 100
        },
        'Hubble': {'z': [0., z]},
        'angular_diameter_distance': {'z': [z]},
        'fsigma8': {'z': [z]},
        'sigma8_z': {'z': [z]},
        'rdrag': None,
    })
    model.logpost(params_dict)
    a = CobayaCambProvider(model.provider, z)
    b = CambProvider(**params_dict, z=z)
    yield (a, z)
    yield (b, z)


def test_CobayaCambProvider(compare_ndarrays):
    dcts = []
    for provider, z in two_providers():
        kh = np.logspace(-4, 0, 200)
        dct = {
            'rdrag': provider.get_rdrag(),
            'h0': provider.get_h0(),
            'DA': provider.get_angular_diameter_distance(z),
            'Hubble': provider.get_Hubble(z),
            'fsigma8': provider.get_fsigma8(z),
            'sigma8': provider.get_sigma8_z(z),
            'pkh': provider.interp_pkh(kh)
        }
        dcts.append(dct)
    ref, data = dcts
    compare_ndarrays(
        ref, data,
        default_tolerance={'atol': 0., 'rtol': 1e-6},
        tolerances={
            'pkh': {'atol': 0., 'rtol': 2e-4},
            'fsigma8': {'atol': 0, 'rtol': 1e-4},
            'sigma8': {'atol': 0, 'rtol': 1e-4}
        },
    )


@pytest.mark.fcompare
def test_compare_camb_and_classy(
    compare_ndarrays, rtol: float, atol: float, yamlroot: Path
):
    camb_model = cobaya.get_model(yamlroot / 'camb_planck18.yaml')
    classy_model = cobaya.get_model(yamlroot / 'classy_planck18.yaml')

    z = 0.5

    camb_model.add_requirements({
        "Pk_grid": {
            "nonlinear": False,
            "z": [z],
            "k_max": 5,
        },
        "Hubble": {'z': [0., z]},
        "angular_diameter_distance": {'z': [z]},
        "fsigma8": {'z': [z]},
        "sigma8_z": {'z': [z]},
        "rdrag": None,
        'sigma8': None,
    })
    classy_model.add_requirements({
        "Pk_interpolator": {
            "nonlinear": False,
            "z": [z],
            "k_max": 5,
        },
        "Hubble": {'z': [0., z]},
        "angular_diameter_distance": {'z': [z]},
        "rdrag": None,
        'sigma8': None,
    })

    camb_point = {
        "logA": 3.05,
        "ombh2": 0.02214,
        "omch2": 0.12,
    }
    classy_point = {
        "logA": 3.05,
        "omega_b": 0.02214,
        "omega_cdm": 0.12,
    }

    camb_model.logpost(camb_point)
    classy_model.logpost(classy_point)

    camb_provider = CobayaCambProvider(camb_model.provider, z)
    classy_provider = CobayaClassyProvider(classy_model.provider, z)

    dcts = []
    for provider in (classy_provider, camb_provider):
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
