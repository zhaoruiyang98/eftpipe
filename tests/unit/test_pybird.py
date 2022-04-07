import numpy as np
from eftpipe import pybird


def test_cosmo_helper_func():
    omegam = 0.2
    z = 1.0
    hubble = 1.549193338482967
    da = 0.4117451980802465
    assert np.isclose(pybird.Hubble(omegam, z), hubble, atol=0.)
    assert np.isclose(pybird.DAfunc(omegam, z), da, atol=0.)
