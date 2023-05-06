from __future__ import annotations
import numpy as np
from pathlib import Path


def main():
    z1_NGC = "patchy/pk_patchy{}_ngc_z1_fkp.txt"
    z1_SGC = "patchy/pk_patchy{}_sgc_z1_fkp.txt"
    z3_NGC = "patchy/pk_patchy{}_ngc_z3_fkp.txt"
    z3_SGC = "patchy/pk_patchy{}_sgc_z3_fkp.txt"
    for name, template in zip(
        ["ngc_z1", "sgc_z1", "ngc_z3", "sgc_z3"],
        [z1_NGC, z1_SGC, z3_NGC, z3_SGC],
    ):
        shape = np.loadtxt(template.format(1)).shape
        data = np.zeros((2048, *shape))
        for i in range(2048):
            data[i] = np.loadtxt(template.format(i + 1))
        # mean = np.mean(data, axis=0)
        stacked = np.zeros((2048, shape[0] * 3))
        for i in range(2048):
            stacked[i] = np.hstack((data[i, :, 1], data[i, :, 2], data[i, :, 3]))
        cov = np.cov(stacked, rowvar=False)
        np.savetxt("cov_" + name + ".dat", cov)


if __name__ == "__main__":
    main()
