from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from eftpipe.tools import timer
from typing import Dict, Generic, TypeVar

ROOT = Path("~/share/eBOSS_DR16").expanduser()


@dataclass
class Info:
    ndata: int
    wndata: int
    norm: float  # without alpha


_K = TypeVar("_K")
_V = TypeVar("_V")


class LoggedDict(Dict[_K, _V]):
    def __setitem__(self, key: _K, value: _V):
        print(f"{key}: {value}")
        super().__setitem__(key, value)


def get_info(path, zmin: float, zmax: float):
    data = np.load(path)["data"]
    mask = (data[:, 2] >= zmin) & (data[:, 2] <= zmax)
    data = data[mask]
    return Info(
        ndata=data.shape[0],
        wndata=data[:, 3].sum(),
        norm=(data[:, 3] ** 2 * data[:, -1]).sum(),
    )


def compute_complete():
    Folder = ROOT / "mock_complete"
    results: LoggedDict[str, float] = LoggedDict()
    file_template = "EZmock_eBOSS_{}_{}_{:0>4}.{}.npz"
    for tracer in ["LRGpCMASS", "ELG"]:
        for cap in ["NGC", "SGC"]:
            zmin = 0.6
            zmax = 1.1 if tracer == "ELG" else 1.0
            dat = [
                get_info(
                    Folder / file_template.format(tracer, cap, i, "dat"), zmin, zmax
                )
                for i in range(1, 1000 + 1)
            ]
            ran_noric = get_info(
                Folder / f"EZmock_eBOSS_{tracer}_{cap}.ran.npz", zmin, zmax
            )
            ran_ric = [
                get_info(
                    Folder / file_template.format(tracer, cap, i, "ran"), zmin, zmax
                )
                for i in range(1, 1000 + 1)
            ]
            alpha_noric = [d.wndata / ran_noric.wndata for d in dat]
            alpha_ric = [d.wndata / r.wndata for d, r in zip(dat, ran_ric)]
            norm_noric = [alpha * ran_noric.norm for alpha in alpha_noric]
            norm_ric = [alpha * r.norm for alpha, r in zip(alpha_ric, ran_ric)]
            results[f"{tracer}.{cap}.noric"] = np.mean(norm_noric)
            results[f"{tracer}.{cap}.noric.std"] = np.std(norm_noric)
            results[f"{tracer}.{cap}.ric"] = np.mean(norm_ric)
            results[f"{tracer}.{cap}.ric.std"] = np.std(norm_ric)


def compute_standard():
    Folder = ROOT / "mock"
    results: LoggedDict[str, float] = LoggedDict()
    file_template = "EZmock_eBOSS_{}_{}_v7_{:0>4}.{}.npz"
    for tracer in ["LRGpCMASS", "ELG"]:
        for cap in ["NGC", "SGC"]:
            zmin = 0.6
            zmax = 1.1 if tracer == "ELG" else 1.0
            dat = [
                get_info(
                    Folder / file_template.format(tracer, cap, i, "dat"), zmin, zmax
                )
                for i in range(1, 1000 + 1)
            ]
            ran = [
                get_info(
                    Folder / file_template.format(tracer, cap, i, "ran"), zmin, zmax
                )
                for i in range(1, 1000 + 1)
            ]
            alpha = [d.wndata / r.wndata for d, r in zip(dat, ran)]
            norm = [a * r.norm for a, r in zip(alpha, ran)]
            results[f"{tracer}.{cap}.standard"] = np.mean(norm)
            results[f"{tracer}.{cap}.standard.std"] = np.std(norm)


def compute():
    Folder = Path("~/source/project/parity/data").expanduser()
    results: LoggedDict[str, float] = LoggedDict()
    file_template = "eBOSS_{}_clustering_{}_vDR16.{}.npz"

    for tracer in ["LRGpCMASS", "ELG"]:
        for cap in ["NGC", "SGC"]:
            zmin = 0.6
            zmax = 1.1 if tracer == "ELG" else 1.0
            dat = [
                get_info(Folder / file_template.format(tracer, cap, "dat"), zmin, zmax)
                for i in range(1, 1 + 1)
            ]
            ran = [
                get_info(Folder / file_template.format(tracer, cap, "ran"), zmin, zmax)
                for i in range(1, 1 + 1)
            ]
            alpha = [d.wndata / r.wndata for d, r in zip(dat, ran)]
            norm = [a * r.norm for a, r in zip(alpha, ran)]
            results[f"{tracer}.{cap}"] = np.mean(norm)


def main():
    compute_complete()
    compute_standard()
    # compute()


if __name__ == "__main__":
    main()
