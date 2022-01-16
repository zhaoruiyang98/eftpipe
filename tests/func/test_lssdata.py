import numpy as np
from eftpipe.lssdata import FullShapeDataParser


def test_FullShapeDataParser():
    dct = {
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
        np.linalg.inv(data_obj.cov) * (1.0 / 27.0) *
        (1000 - 82 - 2) / (1000 - 1),
        data_obj.invcov,
        atol=0.,
    )
