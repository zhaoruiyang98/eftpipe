import cobaya
import os
import pytest
from pathlib import Path
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture


class PathContext:
    def __init__(self, path) -> None:
        self.old = Path('.').resolve()
        self.new = self.old / path

    def __enter__(self):
        os.chdir(self.new)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        os.chdir(self.old)


def test_LRG_NGC_reg(ndarrays_regression: NDArraysRegressionFixture):
    yaml_name = Path('.').resolve() / \
        'tests/reg/yamls/mock_LRG_NGC_km0p15_fixref_proposal.yaml'
    with PathContext('cobaya'):
        model = cobaya.get_model(yaml_name)
        params_dict = {
            'omch2': 1.111943844e-01,
            'H0': 6.630797882e+01,
            'logA': 3.129647579e+00,
            'ombh2': 0.02214,
            'ns': 0.9611,
            'mnu': 0.0,
            'tau': 0.0543,
            'LRG_NGC_b1': 2.090642334e+00,
            'LRG_NGC_c2': 6.069025573e-01,
            'LRG_NGC_b2': 4.291449138e-01,
            'LRG_NGC_c4': 0.,
            'LRG_NGC_b3': 4.518014849e-01,
            'LRG_NGC_b4': 4.291449138e-01,
            'LRG_NGC_cct': -1.209669604e+00,
            'LRG_NGC_cr1': -2.907440471e+00,
            'LRG_NGC_cr2': -5.965905957e-01,
            'LRG_NGC_ce0': 2.348928561e-01,
            'LRG_NGC_cemono': 0.0,
            'LRG_NGC_cequad': -8.642105220e-01,
            'LRG_NGC_knl': 0.7,
            'LRG_NGC_km': 0.7,
            'LRG_NGC_nd': 7.91e-05
        }
        sampled_dict = model.parameterization.sampled_params()
        sampled_dict = {key: params_dict[key] for key in sampled_dict}
        logpost = model.logpost(sampled_dict)

        theory_vector = model.likelihood['NGC'].theory_vector(**params_dict)
        data_vector = model.likelihood['NGC'].data_obj.data_vector
        invcov = model.likelihood['NGC'].data_obj.invcov
        reduced_chi2 = model.likelihood['NGC'].get_param('reduced_chi2')
        reg_dct = {
            'theory_vector': theory_vector,
            'data_vector': data_vector,
            'invcov': invcov,
            'reduced_chi2': reduced_chi2
        }
        ndarrays_regression.check(
            reg_dct, default_tolerance={'atol': 0, 'rtol': 1e-8}
        )


def test_ELG_NGC_reg(ndarrays_regression: NDArraysRegressionFixture):
    yaml_name = Path('.').resolve() / \
        'tests/reg/yamls/mock_ELG_NGC_km0p15_Q_fixref_proposal.yaml'
    with PathContext('cobaya'):
        model = cobaya.get_model(yaml_name)
        params_dict = {
            'omch2': 1.119507559e-01,
            'H0': 6.770409490e+1,
            'logA': 2.815516798e+00,
            'ombh2': 0.02214,
            'ns': 0.9611,
            'mnu': 0.0,
            'tau': 0.0543,
            'ELG_NGC_b1': 1.589354637e+00,
            'ELG_NGC_c2': 1.266786101e+00,
            'ELG_NGC_b2': 8.957530420e-01,
            'ELG_NGC_c4': 0.,
            'ELG_NGC_b3': 1.675518719e-01,
            'ELG_NGC_b4': 8.957530420e-01,
            'ELG_NGC_cct': -7.109794839e-01,
            'ELG_NGC_cr1': -2.635512354e+00,
            'ELG_NGC_cr2': -3.667395306e-01,
            'ELG_NGC_ce0': 1.634998078e-02,
            'ELG_NGC_cemono': 0.0,
            'ELG_NGC_cequad': -4.125790673e-02,
            'ELG_NGC_knl': 0.45,
            'ELG_NGC_km': 0.45,
            'ELG_NGC_nd': 1.851851852e-04
        }
        sampled_dict = model.parameterization.sampled_params()
        sampled_dict = {key: params_dict[key] for key in sampled_dict}
        logpost = model.logpost(sampled_dict)

        theory_vector = model.likelihood['NGC'].theory_vector(**params_dict)
        data_vector = model.likelihood['NGC'].data_obj.data_vector
        invcov = model.likelihood['NGC'].data_obj.invcov
        reduced_chi2 = model.likelihood['NGC'].get_param('reduced_chi2')
        reg_dct = {
            'theory_vector': theory_vector,
            'data_vector': data_vector,
            'invcov': invcov,
            'reduced_chi2': reduced_chi2
        }
        ndarrays_regression.check(
            reg_dct, default_tolerance={'atol': 0, 'rtol': 1e-8}
        )


def test_LRG_ELG_x_NGC_reg(ndarrays_regression: NDArraysRegressionFixture):
    yaml_name = Path('.').resolve() / \
        'tests/reg/yamls/mock_LRG_ELG_x_NGC_km0p15_fix_proposal.yaml'
    with PathContext('cobaya'):
        model = cobaya.get_model(yaml_name)
        params_dict = {
            'omch2': 1.1882587E-01,
            'H0': 6.7693170E+01,
            'logA': 2.8304142E+00,
            'ombh2': 0.02214,
            'ns': 0.9611,
            'mnu': 0.0,
            'tau': 0.0543,
            'LRG_NGC_b1': 2.4088606E+00,
            'LRG_NGC_c2': 1.4996826E+00,
            'LRG_NGC_b2': 1.0604357E+00,
            'LRG_NGC_c4': 0.,
            'LRG_NGC_b3': -1.4869963E-02,
            'LRG_NGC_b4': 1.0604357E+00,
            'LRG_NGC_cct': -8.4508179E-01,
            'LRG_NGC_cr1': -6.8305956E+00,
            'LRG_NGC_cr2': -9.9569434E-01,
            'LRG_NGC_ce0': 2.7224520E-02,
            'LRG_NGC_cemono': 0.0,
            'LRG_NGC_cequad': -2.1169575E-01,
            'LRG_NGC_knl': 0.7,
            'LRG_NGC_km': 0.7,
            'LRG_NGC_nd': 7.91e-05,
            'ELG_NGC_b1': 1.4872198E+00,
            'ELG_NGC_c2': 1.0307483E+00,
            'ELG_NGC_b2': 7.2884914E-01,
            'ELG_NGC_c4': 0.,
            'ELG_NGC_b3': 3.2409937E-01,
            'ELG_NGC_b4': 7.2884914E-01,
            'ELG_NGC_cct': -4.3884597E-01,
            'ELG_NGC_cr1': -1.7839895E+00,
            'ELG_NGC_cr2': -3.7603389E-01,
            'ELG_NGC_ce0': 3.3194907E-02,
            'ELG_NGC_cemono': 0.0,
            'ELG_NGC_cequad': -1.0867432E-01,
            'ELG_NGC_knl': 0.45,
            'ELG_NGC_km': 0.45,
            'ELG_NGC_nd': 0.00018518518518518518,
            'x_NGC_ce0': -8.5830850E-02,
            'x_NGC_cemono': 0.0,
            'x_NGC_cequad': -1.0641260E-01,
            'x_NGC_km': 0.7,
        }
        sampled_dict = model.parameterization.sampled_params()
        sampled_dict = {key: params_dict[key] for key in sampled_dict}
        logpost = model.logpost(sampled_dict)

        theory_vector = model.likelihood['NGC'].theory_vector(**params_dict)
        data_vector = model.likelihood['NGC'].data_obj.data_vector
        invcov = model.likelihood['NGC'].data_obj.invcov
        reduced_chi2 = model.likelihood['NGC'].get_param('reduced_chi2')
        reg_dct = {
            'theory_vector': theory_vector,
            'data_vector': data_vector,
            'invcov': invcov,
            'reduced_chi2': reduced_chi2
        }
        ndarrays_regression.check(
            reg_dct, default_tolerance={'atol': 0, 'rtol': 1e-8}
        )
