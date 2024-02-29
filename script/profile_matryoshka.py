import numpy as np
import matryoshka.emulator as matry
from astropy.cosmology import Planck18


def main():
    COSMO = np.array(
        [
            Planck18.Om0,
            Planck18.Ob0,
            Planck18.H0.value / 100.0,
            Planck18.Neff,
            -1.0,
        ]
    )
    TRANSFER = matry.Transfer()

    for i in range(1000):
        _ = TRANSFER.emu_predict(COSMO, mean_or_full="mean")


if __name__ == "__main__":
    main()
