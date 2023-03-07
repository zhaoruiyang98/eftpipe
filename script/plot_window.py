from __future__ import annotations
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from eftpipe.pybird.pybird import Common, kbird, Window, window_kgrid


def plot_window(
    axes,
    k: float,
    window,
    hex=False,
    kbird=kbird,
    xlim=None,
    ylim=None,
    accboost: int = 1,
):
    i = np.searchsorted(kbird, k)
    p = window_kgrid(accboost=accboost)
    axes.plot(p, window[0, 0, i, :], ls="-", c="k", label=r"$W_{00}$")
    axes.plot(p, window[0, 1, i, :], ls="--", c="k", label=r"$W_{02}$")
    axes.plot(p, window[1, 0, i, :], ls="-", c="b", label=r"$W_{20}$")
    axes.plot(p, window[1, 1, i, :], ls="--", c="b", label=r"$W_{22}$")
    if hex:
        axes.plot(p, window[0, 2, i, :], ls=":", c="k", label=r"$W_{04}$")
        axes.plot(p, window[1, 2, i, :], ls=":", c="b", label=r"$W_{24}$")
        if window.shape[0] >= 3:
            axes.plot(p, window[2, 0, i, :], ls="-", c="g", label=r"$W_{40}$")
            axes.plot(p, window[2, 1, i, :], ls="--", c="g", label=r"$W_{42}$")
            axes.plot(p, window[2, 2, i, :], ls=":", c="g", label=r"$W_{44}$")
    axes.axhline(y=0.0, lw=1, c="c")
    axes.set_xlabel(R"$k'\ [h\;\mathrm{Mpc}^{-1}]$")
    axes.set_ylabel(R"$W(k,k')_{\ell,\ell'}\ [h\;\mathrm{Mpc}^{-1}]$")
    axes.legend(frameon=False)
    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)
    axes.set_title(Rf"$k={kbird[i]}$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("window", type=Path, help="window file")
    parser.add_argument("k", type=float, help="k")
    parser.add_argument("--hex", action="store_true", help="plot hexadecapole")
    parser.add_argument("-xmin", type=float, default=None, help="xmin")
    parser.add_argument("-xmax", type=float, default=None, help="xmax")
    parser.add_argument("-ymin", type=float, default=None, help="ymin")
    parser.add_argument("-ymax", type=float, default=None, help="ymax")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--regen",
        nargs="+",
        default=None,
        help="pass arg=value sequence to regenerate window",
    )
    group.add_argument(
        "--icc", action="store_true", help="extract window from icc file"
    )
    parser.add_argument("--PSN", action="store_true", help="plot PSN from icc file")
    args = parser.parse_args()

    plt.rcParams["text.usetex"] = True
    if not args.regen:
        window = np.load(args.window)
        with args.window.with_suffix(".json").open() as f:
            meta = json.load(f)
        plot_window(
            plt.gca(),
            args.k,
            window["Wal"] if args.icc else window,
            args.hex,
            xlim=(args.xmin, args.xmax),
            ylim=(args.ymin, args.ymax),
            accboost=meta["accboost"],
        )
        plt.show()
        if args.icc and args.PSN:
            # XXX: may not be correct
            k = kbird
            PSN = window["PSN"]
            fn = interp1d(k, PSN, kind="cubic", axis=-1)
            k = np.geomspace(k[0], k[-1], 1000)
            PSN = fn(k)
            plt.semilogx(k, PSN[0], c="k", label=R"$\ell=0$")
            plt.semilogx(k, PSN[1], c="b", label=R"$\ell=2$")
            if args.hex:
                plt.semilogx(k, PSN[2], c="g", label=R"$\ell=4$")
            plt.title(R"$P_\mathrm{SN}$")
            plt.show()

        parser.exit()

    override = {}
    for arg in args.regen:
        key, value = arg.split("=")
        if "." in value:
            value = float(value)
        elif "'" in value:
            value = value
        else:
            value = int(value)
        override[key] = value

    with args.window.with_suffix(".json").open() as f:
        meta = json.load(f)
    meta.pop("k")
    co = Common(Nl=override.get("Nl", meta["Nl"]))
    override.update({"co": co, "load": False, "save": False})
    meta.update(override)
    tmp = Window(**meta)
    window = tmp.Wal
    plot_window(
        plt.gca(),
        args.k,
        window,
        args.hex,
        xlim=(args.xmin, args.xmax),
        ylim=(args.ymin, args.ymax),
        accboost=tmp.meta["accboost"],
    )
    plt.show()


if __name__ == "__main__":
    main()
