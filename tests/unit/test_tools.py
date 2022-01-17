import yaml
from pathlib import Path
from eftpipe.tools import update_path_in_dict


def test_update_path_in_dict():
    info = r"""
    path: mock
    path2:
    - x: cov
    - path_y: cov
    extra_args:
        data:
            pklinfo:
            - pkl_path: data/mock/LRG_NGC.txt
            - pkl_path: data/mock/ELG_NGC_Q.txt
            common:
                ls: [0, 2]
            cov_path: data/cov/cov_NGC_L024E024X024_PQP.txt
        theory:
            theory_info:
            - z: 0.7
              projection_config:
                  windows_fourier_path: cache/LRG.npy
            - z: 0.845
              projection_config:
                  windows_fourier_path: cache/ELG.npy
    """
    dct = yaml.load(info, Loader=yaml.SafeLoader)
    update_path_in_dict(dct, Path('test'))
    def join(s):
        return str(Path('test') / s)
    true_dct = {
        "path": join('mock'),
        "path2": [{"x": 'cov'}, {"path_y": join('cov')}],
        "extra_args": {
            "data": {
                "pklinfo": [
                    {"pkl_path": join("data/mock/LRG_NGC.txt")},
                    {"pkl_path": join("data/mock/ELG_NGC_Q.txt")}
                ],
                "common": {"ls": [0, 2]},
                "cov_path": join("data/cov/cov_NGC_L024E024X024_PQP.txt")
            },
            "theory": {
                "theory_info": [
                    {
                        "z": 0.7,
                        "projection_config": {
                            "windows_fourier_path": join("cache/LRG.npy")
                        }
                    },
                    {
                        "z": 0.845,
                        "projection_config": {
                            "windows_fourier_path": join("cache/ELG.npy")
                        }
                    }
                ]
            }
        }
    }
    assert dct == true_dct
