from __future__ import annotations

import enum
import numpy as np
from pathlib import Path

DATA_ROOT = Path(__file__).parent
NORIC_ROOT = DATA_ROOT / "raw" / "pk_measurement" / "noric"


class Lcol(enum.IntEnum):
    """column number for LRG"""

    K = 0
    L0 = 1
    L2 = 3
    L4 = 5


class Ecol(enum.IntEnum):
    """column number for ELG"""

    K = 0
    L0 = 8
    L2 = 10
    L4 = 12


class Xcol(enum.IntEnum):
    """column number for cross"""

    K = 0
    L0 = 15
    L2 = 17
    L4 = 19


def main():
    for cap in ("NGC", "SGC"):
        data_PPP = np.loadtxt(NORIC_ROOT / f"pk_PPP_mean_sigma_{cap}.dat").T
        data_QQP = np.loadtxt(NORIC_ROOT / f"pk_QQP_mean_sigma_{cap}.dat").T
        for tracer, Col in zip(("LRG", "ELG", "x"), (Lcol, Ecol, Xcol)):
            for is_Q, data in zip((True, False), (data_QQP, data_PPP)):
                if tracer == "x" and is_Q:
                    continue
                suffix = "_Q" if is_Q else ""
                save_path = DATA_ROOT / "mock" / f"noric_{tracer}_{cap}{suffix}.txt"
                ks = data[Col.K]
                columns = [ks, data[Col.L0] / ks, data[Col.L2] / ks, data[Col.L4] / ks]
                out = np.vstack(columns).T
                if is_Q:
                    header = (5 * " ").join(["k", "Q0", "Q2", "Q4"])
                else:
                    header = (5 * " ").join(["k", "P0", "P2", "P4"])
                np.savetxt(save_path, out, header=header)

        # order: LRG ELG X
        cov_PPP = np.loadtxt(NORIC_ROOT / f"cov_PPP_{cap}.dat")
        cov_PQP = np.loadtxt(NORIC_ROOT / f"cov_PQP_{cap}.dat")
        cov_QQP = np.loadtxt(NORIC_ROOT / f"cov_QQP_{cap}.dat")
        nlen = cov_PPP.shape[0] // 3
        # single tracer
        for tracer, ix in zip(
            ("LRG", "ELG", "X"),
            (slice(0, nlen), slice(nlen, 2 * nlen), slice(2 * nlen, 3 * nlen)),
        ):
            for is_Q, cov in zip((True, False), (cov_QQP, cov_PPP)):
                if tracer == "X" and is_Q:
                    continue
                suffix = "_Q" if is_Q else ""
                save_path = (
                    DATA_ROOT / "cov" / f"noric_cov_{cap}_{tracer[0]}024{suffix}.txt"
                )
                np.savetxt(save_path, cov[ix, ix])
        # two tracer + cross
        for suffix, cov in zip(("", "_PQP", "_QQP"), (cov_PPP, cov_PQP, cov_QQP)):
            save_path = DATA_ROOT / "cov" / f"noric_cov_{cap}_L024E024X024{suffix}.txt"
            np.savetxt(save_path, cov)
        # two tracer
        for suffix, cov in zip(("", "_PQ", "_QQ"), (cov_PPP, cov_PQP, cov_QQP),):
            save_path = DATA_ROOT / "cov" / f"noric_cov_{cap}_L024E024{suffix}.txt"
            np.savetxt(save_path, cov[: 2 * nlen, : 2 * nlen])


if __name__ == "__main__":
    main()
