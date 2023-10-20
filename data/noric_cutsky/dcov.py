from __future__ import annotations
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
SAVEROOT = ROOT
single = "cov_{cap}_{t}024_P.txt"
double = "cov_{cap}_L024E024_PP.txt"
triple = "cov_{cap}_L024E024X024_PPP.txt"


def copy_block_diag(arr, val, nblock):
    nvec = arr.shape[0] // nblock
    for i in range(nblock):
        for j in range(nblock):
            istart, iend = i * nvec, (i + 1) * nvec
            jstart, jend = j * nvec, (j + 1) * nvec
            from_ = val[istart:iend, jstart:jend]
            to_ = arr[istart:iend, jstart:jend]
            np.fill_diagonal(to_, np.diagonal(from_))


for cap in ["NGC", "SGC"]:
    for t in ["L", "E", "X"]:
        cov = np.loadtxt(ROOT / single.format(cap=cap, t=t))
        tosave = np.zeros_like(cov)
        copy_block_diag(tosave, cov, 3)
        np.savetxt(SAVEROOT / ("d" + single.format(cap=cap, t=t)), tosave)

    cov = np.loadtxt(ROOT / double.format(cap=cap))
    tosave = np.zeros_like(cov)
    copy_block_diag(tosave, cov, 6)
    np.savetxt(SAVEROOT / ("d" + double.format(cap=cap)), tosave)

    cov = np.loadtxt(ROOT / triple.format(cap=cap))
    tosave = np.zeros_like(cov)
    copy_block_diag(tosave, cov, 9)
    np.savetxt(SAVEROOT / ("d" + triple.format(cap=cap)), tosave)
