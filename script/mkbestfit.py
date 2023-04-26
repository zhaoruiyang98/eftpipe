from __future__ import annotations
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pprint import pprint
from typing import cast, Sequence
from eftpipe.analysis import BestfitModel
from eftpipe.tools import PathContext

plt.rcParams["text.usetex"] = True


class TypedNamespace(argparse.Namespace):
    input_yaml: Path
    tracers: list[str]
    output: str | None
    verbose: bool
    base: Path
    sharey: bool
    ymin: float | None
    ymax: float | None
    kmin: float | None
    kmax: float | None
    nrows: int
    freeze: list[tuple[str, str]]
    freeze_binned: bool


def get_argparser():
    parser = argparse.ArgumentParser(
        "mkbestfit", description="Plot best fit theory curve (w/o binning)"
    )
    parser.add_argument("input_yaml", type=Path)
    parser.add_argument(
        "--tracers",
        nargs="+",
        required=True,
        metavar="TRACER",
        help="tracers to plot, 'all' for all tracers",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="output file name, default is to show"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show logging")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path.cwd(),
        help="working path when loading yaml file (default cwd)",
    )
    # plot style
    parser.add_argument(
        "--sharey", action="store_true", help="share ylim for all subplots"
    )
    parser.add_argument(
        "--ymin", type=float, help="set the minimum of y axis for all subplots"
    )
    parser.add_argument(
        "--ymax", type=float, help="set the maximum of y axis for all subplots"
    )
    parser.add_argument(
        "--kmin", type=float, help="set the minimum of x axis for all subplots"
    )
    parser.add_argument(
        "--kmax", type=float, help="set the maximum of x axis for all subplots"
    )
    parser.add_argument(
        "--nrows", type=int, default=1, help="number of rows in the plot (default 1)"
    )
    # freeze theory
    parser.add_argument(
        "--freeze",
        nargs=2,
        action="append",
        metavar=("TRACER", "FILE"),
        default=[],
        help="freeze TRACER's theoretical output (w/o binning) to FILE.",
    )
    parser.add_argument(
        "--freeze-binned",
        action="store_true",
        help="freeze binned theory instead of unbinned theory",
    )
    return parser


def main(argv: Sequence[str] | None = None):
    parser = get_argparser()
    args = cast(TypedNamespace, parser.parse_args(argv))
    if args.freeze_binned:
        parser.exit(message="freeze_binned not implemented yet!\n")
    input_yaml = str(args.input_yaml.expanduser().resolve())
    with PathContext(args.base):
        model = BestfitModel(input_yaml, verbose=args.verbose)
    print("bestfit:")
    pprint(model.bestfit)
    print("fullchi2 (w/o hartlap):")
    for k, v in model.fullchi2.items():
        print(f"{k}={v}")

    tracers = args.tracers
    if tracers[0] == "all":
        tracers = model.tracers.copy()
        if tracers == ["LRG_NGC", "ELG_NGC", "X_NGC", "LRG_SGC", "ELG_SGC", "X_SGC"]:
            tracers = ["LRG_NGC", "X_NGC", "ELG_NGC", "LRG_SGC", "X_SGC", "ELG_SGC"]
        elif tracers == ["LRG_NGC", "ELG_NGC", "X_NGC"]:
            tracers = ["LRG_NGC", "X_NGC", "ELG_NGC"]
        elif tracers == ["LRG_SGC", "ELG_SGC", "X_SGC"]:
            tracers = ["LRG_SGC", "X_SGC", "ELG_SGC"]
    nrows = args.nrows
    ncols = math.ceil(len(tracers) / nrows)
    width = 5.5 * ncols
    height = 4 * nrows
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, height), sharey=args.sharey, sharex=True
    )
    for i, tracer in enumerate(tracers):
        ax = plt.subplot(nrows, ncols, i + 1)
        model.plot(tracer, ax=ax)
        if args.ymin is not None:
            ax.set_ylim(ymin=args.ymin)
        if args.ymax is not None:
            ax.set_ylim(ymax=args.ymax)
        if args.kmin is not None:
            ax.set_xlim(xmin=args.kmin)
        if args.kmax is not None:
            ax.set_xlim(xmax=args.kmax)
    fig.tight_layout()
    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output)

    for tracer, file in args.freeze:
        k = model.multipoles[tracer].k
        symbol = model.multipoles[tracer].symbol
        Plk = model.Plk_interpolator(tracer)
        output = np.vstack([k, *Plk(Plk.ls, k)]).T
        header = (" " * 5).join(["k"] + [f"{symbol}{ell}" for ell in Plk.ls])
        np.savetxt(file, output, header=header)


if __name__ == "__main__":
    main()
