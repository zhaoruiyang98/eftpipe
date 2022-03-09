import numpy as np
import yaml
from eftpipe.lssdata import FullShapeDataDict
from eftpipe.parser import (
    FullShapeDataParser,
    SingleTracerParser,
    TwoTracerParser,
    TwoTracerCrossParser,
)


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
    dct = {
        'data': {
            'pklinfo': {
                'ls': [0, 2],
                'kmin': 0.02,
                'kmax': 0.15,
                'pkl_path': 'cobaya/data/mock/LRG_NGC.txt'
            },
            'cov_path': 'cobaya/data/cov/cov_NGC_L024.txt',
            'Nreal': 1000,
        },
        'theory': {
            'prefix': 'LRG_NGC_',
            'Nl': 2,
            'z': 0.7,
            'kmA': 0.7,
            'ndA': 7.91e-05,
            'cache_dir_path': 'cobaya/cache',
            'projection_config': {
                'Om_AP': 0.307115,
                'z_AP': 0.7,
                'rdrag_fid': 147.66,
                'windows_fourier_path': 'cobaya/cache/LRG_NGC_GB_interp.npy',
                'windows_configspace_path': 'cobaya/data/window/LRG_NGC_GB_interp.dat',
                'binning': True,
            }
        }
    }
    parser = SingleTracerParser(dct)
    data_obj = parser.create_gaussian_data()
    theory_obj = parser.create_vector_theory()


def test_TwoTracerParser():
    info = r"""
    data:
        cov_path: cobaya/data/cov/cov_NGC_L024E024_PQ.txt
        Nreal: 1000
        rescale: 1.0
        pklinfo:
            - pkl_path: cobaya/data/mock/LRG_NGC.txt
            - pkl_path: cobaya/data/mock/ELG_NGC_Q.txt
        common:
            ls: [0, 2]
            kmin: 0.02
            kmax: 0.15
    theory:
        prefix: [LRG_NGC_, ELG_NGC_]
        theory_info:
            - z: 0.7
              kmA: 0.7
              ndA: 0.0000791
              projection_config:
                  z_AP: 0.7
                  windows_fourier_path: cobaya/cache/LRG_NGC_GB_interp.npy
                  windows_configspace_path: cobaya/data/window/LRG_NGC_GB_interp.dat
            - z: 0.845
              kmA: 0.45
              ndA: 0.00018518518518518518
              chained: True
              projection_config:
                  z_AP: 0.845
                  windows_fourier_path: cobaya/cache/ELG_NGC_interp.npy
                  windows_configspace_path: cobaya/data/window/ELG_NGC_interp.dat
        common:
            cache_dir_path: cobaya/cache
            Nl: 2
            chained: False
            projection_config:
                Om_AP: 0.307115
                rdrag_fid: 147.66
                binning: true
    """
    dct = yaml.load(info, Loader=yaml.SafeLoader)
    parser = TwoTracerParser(dct)
    data_obj = parser.create_gaussian_data()
    theory_obj = parser.create_vector_theory()


def test_TwoTracerCrossParser():
    info = r"""
    data:
        cov_path: cobaya/data/cov/cov_NGC_L024E024X024_PQP.txt
        Nreal: 1000
        rescale: 1.0
        pklinfo:
            - pkl_path: cobaya/data/mock/LRG_NGC.txt
            - pkl_path: cobaya/data/mock/ELG_NGC_Q.txt
            - pkl_path: cobaya/data/mock/x_NGC.txt
        common:
            ls: [0, 2]
            kmin: 0.02
            kmax: 0.15
    theory:
        prefix: [LRG_NGC_, ELG_NGC_, x_NGC_]
        theory_info:
            - z: 0.7
              kmA: 0.7
              ndA: 0.0000791
              projection_config:
                  z_AP: 0.7
                  windows_fourier_path: cobaya/cache/LRG_NGC_GB_interp.npy
                  windows_configspace_path: cobaya/data/window/LRG_NGC_GB_interp.dat
            - z: 0.845
              kmA: 0.45
              ndA: 0.00018518518518518518
              chained: True
              projection_config:
                  z_AP: 0.845
                  windows_fourier_path: cobaya/cache/ELG_NGC_interp.npy
                  windows_configspace_path: cobaya/data/window/ELG_NGC_interp.dat
            - z: 0.77
              kmA: 0.7
              ndA: 0.0000791
              kmB: 0.45
              ndB: 0.00018518518518518518
              cross: True
              projection_config:
                z_AP: 0.77
                windows_fourier_path: cobaya/cache/x_NGC_interp.npy
                windows_configspace_path: cobaya/data/window/x_NGC_interp.dat
        common:
            cache_dir_path: cobaya/cache
            Nl: 2
            chained: False
            projection_config:
                Om_AP: 0.307115
                rdrag_fid: 147.66
                binning: true
    """
    dct = yaml.load(info, Loader=yaml.SafeLoader)
    parser = TwoTracerCrossParser(dct)
    data_obj = parser.create_gaussian_data()
    theory_obj = parser.create_vector_theory()
