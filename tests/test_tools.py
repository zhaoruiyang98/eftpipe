import pytest
import yaml
from pathlib import Path
from eftpipe.tools import set_value_in_nested_dict
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
                  windows_fourier_file: cache/LRG.npy
            - z: 0.845
              projection_config:
                  windows_fourier_path: cache/ELG.npy
                  windows_configspace_file: test/ELG_window.txt
    """
    dct = yaml.load(info, Loader=yaml.SafeLoader)
    update_path_in_dict(
        dct, Path("test"), extra=("windows_fourier_file", "windows_configspace_file")
    )

    def join(s):
        return str(Path("test") / s)

    true_dct = {
        "path": join("mock"),
        "path2": [{"x": "cov"}, {"path_y": join("cov")}],
        "extra_args": {
            "data": {
                "pklinfo": [
                    {"pkl_path": join("data/mock/LRG_NGC.txt")},
                    {"pkl_path": join("data/mock/ELG_NGC_Q.txt")},
                ],
                "common": {"ls": [0, 2]},
                "cov_path": join("data/cov/cov_NGC_L024E024X024_PQP.txt"),
            },
            "theory": {
                "theory_info": [
                    {
                        "z": 0.7,
                        "projection_config": {
                            "windows_fourier_file": join("cache/LRG.npy")
                        },
                    },
                    {
                        "z": 0.845,
                        "projection_config": {
                            "windows_fourier_path": join("cache/ELG.npy"),
                            "windows_configspace_file": join("test/ELG_window.txt"),
                        },
                    },
                ]
            },
        },
    }
    assert dct == true_dct


dcts = [
    {"z": 0.3},
    {"z": 0.3, "config_settings": None},
    {"z": 0.3, "config_settings": {"with_binning": True}},
    {"z": 0.3, "config_settings": {"with_binning": True, "binning": None}},
    {"z": 0.3, "config_settings": {"with_binning": True, "binning": {"accboost": 1}}},
    {
        "z": 0.3,
        "config_settings": {
            "with_binning": True,
            "binning": {"accboost": 1, "kout": None},
        },
    },
    {
        "z": 0.3,
        "config_settings": {
            "with_binning": True,
            "binning": {"accboost": 1, "kout": [4, 5, 6]},
        },
    },
]

kout = [0, 1, 2]
expect = [
    {"z": 0.3, "config_settings": {"binning": {"kout": kout}}},
    {"z": 0.3, "config_settings": {"binning": {"kout": kout}}},
    {"z": 0.3, "config_settings": {"with_binning": True, "binning": {"kout": kout}}},
    {"z": 0.3, "config_settings": {"with_binning": True, "binning": {"kout": kout}}},
    {
        "z": 0.3,
        "config_settings": {
            "with_binning": True,
            "binning": {"accboost": 1, "kout": kout},
        },
    },
    {
        "z": 0.3,
        "config_settings": {
            "with_binning": True,
            "binning": {"accboost": 1, "kout": kout},
        },
    },
    {
        "z": 0.3,
        "config_settings": {
            "with_binning": True,
            "binning": {"accboost": 1, "kout": kout},
        },
    },
]


@pytest.mark.parametrize("pair", zip(dcts, expect))
def test_set_value_in_nested_dict(pair):
    test, expect = pair
    set_value_in_nested_dict(test, kout, "config_settings", "binning", "kout")
    assert test == expect
