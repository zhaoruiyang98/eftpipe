import numpy as np
from cobaya.yaml import yaml_load
from cobaya.yaml import yaml_load_file
from eftpipe.lssdata import FullShapeDataDict
from eftpipe.parser import FullShapeDataParser
from eftpipe.parser import SingleTracerParser
from eftpipe.parser import TwoTracerCrossParser
from eftpipe.parser import TwoTracerParser
from eftpipe.tools import PathContext


def test_FullShapeDataParser():
    dct: FullShapeDataDict = {
        'pklinfo': [
            {'pkl_path': 'cobaya/data/mock/LRG_NGC.txt'},
            {'pkl_path': 'cobaya/data/mock/ELG_NGC_Q.txt',
             'kmax': 0.20},
            {'pkl_path': 'cobaya/data/mock/x_NGC.txt',
             'kmin': 0.05},
        ],
        'common': {
            'ls': [0, 2],
            'kmin': 0.02,
            'kmax': 0.15,
        },
        'cov_path': 'cobaya/data/cov/cov_NGC_L024E024X024_PQP.txt',
        'Nreal': 1000,
        'rescale': 1.0 / 27.0,
    }
    parser = FullShapeDataParser(dct)
    data_obj = parser.create_gaussian_data()
    pkldatas = data_obj.pkldatas
    assert (pkldatas[0].kdata[0] >= 0.02) and (pkldatas[0].kdata[-1] <= 0.15)
    assert (pkldatas[1].kdata[0] >= 0.02) and (pkldatas[1].kdata[-1] <= 0.20)
    assert (pkldatas[2].kdata[0] >= 0.05) and (pkldatas[2].kdata[-1] <= 0.15)
    assert np.allclose(
        np.linalg.inv(data_obj.cov) *
        (1000 - 82 - 2) / (1000 - 1),
        data_obj.invcov,
        atol=0.,
    )


def test_SingleTracerParser():
    dct = yaml_load_file("tests/yamls/mock_LRG_NGC_km0p15_fixref_proposal.yaml")
    dct = dct["likelihood"]["NGC"]["extra_args"]
    with PathContext("cobaya"):
        parser = SingleTracerParser(dct)
        data_obj = parser.create_gaussian_data()
        theory_obj = parser.create_vector_theory()


def test_TwoTracerParser():
    info = r"""
    data:
      pklinfo:
      - pkl_path: data/mock/LRG_NGC.txt
      - pkl_path: data/mock/ELG_NGC_Q.txt
      common:
        ls: [0, 2]
        kmin: 0.02
        kmax: 0.15
      cov_path: data/cov/cov_NGC_L024E024_PQ.txt
      Nreal: 1000
    theory:
        prefix: [LRG_NGC_, ELG_NGC_]
        theory_info:
        - z: 0.7
          km: 0.7
          nd: 7.91e-05
          config_settings:
            APeffect:
              z_AP: 0.7
            window:
              window_fourier_file: cache/LRG_NGC_GB_interp.npy
              window_configspace_file: data/window/LRG_NGC_GB_interp.dat
        - z: 0.845
          km: 0.45
          nd: 0.00018518518518518518
          chained: true
          config_settings:
            APeffect:
              z_AP: 0.845
            window:
              window_fourier_file: cache/ELG_NGC_interp.npy
              window_configspace_file: data/window/ELG_NGC_interp.dat
        common:
          cache_dir_path: cache
          with_IRresum: true
          with_APeffect: true
          with_window: true
          with_fiber: false
          with_binning: true
          config_settings:
            APeffect:
              Om_AP: 0.307115
            binning:
              binning: true
    """
    dct = yaml_load(info)
    with PathContext("cobaya"):
        parser = TwoTracerParser(dct)
        data_obj = parser.create_gaussian_data()
        theory_obj = parser.create_vector_theory()


def test_TwoTracerCrossParser():
    dct = yaml_load_file("tests/yamls/mock_LRG_ELG_x_NGC_km0p15_fix_proposal.yaml")
    dct = dct["likelihood"]["NGC"]["extra_args"]
    with PathContext("cobaya"):
        parser = TwoTracerCrossParser(dct)
        data_obj = parser.create_gaussian_data()
        theory_obj = parser.create_vector_theory()
