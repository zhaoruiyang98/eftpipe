import numpy as np
from eftpipe.pybird.pybird import FFTLog


def FFTLog_results():
    klim = np.logspace(-4, 0, 200)
    pklim = np.exp(-((np.log(klim) - np.log(klim[klim.size // 2])) ** 2))
    pklim_vector = np.vstack([pklim for _ in range(500)])
    fft = FFTLog(Nmax=256, xmin=10 ** -5, xmax=10, bias=-0.3)

    base_result = [
        fft.Coef(klim, x, extrap="padding", window=0.3) for x in pklim_vector
    ]
    base_dct = {"coef": np.vstack(base_result)}
    yield base_dct
    yield {"coef": fft.Coef(klim, pklim_vector, extrap="padding", window=0.3)}


def test_FFTLog_vectorization(compare_ndarrays):
    ref, data = FFTLog_results()
    compare_ndarrays(
        ref, data, default_tolerance={"atol": 0.0, "rtol": 1e-6},
    )
