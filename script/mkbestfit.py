# NOTE: this script has not been well tested
from __future__ import annotations
import argparse
import itertools
import os
import pprint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.cm import tab10  # type: ignore
from typing import Any, TYPE_CHECKING, Sequence
from cobaya import get_model
from cobaya.yaml import yaml_dump
from cobaya.yaml import yaml_load_file
from eftpipe.likelihood import EFTLike
from eftpipe.likelihood import EFTLikeSingle
from eftpipe.likelihood import EFTLikeDouble
from eftpipe.likelihood import EFTLikeDoubleCross
from eftpipe.likelihood import extract_multipole_info
from eftpipe.likelihood import find_covariance_reader
from eftpipe.tools import do_nothing
from eftpipe.tools import disable_logging
from eftpipe.tools import NOTFOUND
from eftpipe.tools import PathContext

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from cobaya.model import Model
    from eftpipe.likelihood import MultipoleInfo
    from eftpipe.theory import PlkInterpolator

GRAY = tab10.colors[-3]
BLUE = tab10.colors[0]
ORANGE = tab10.colors[1]
GREEN = tab10.colors[2]
RED = tab10.colors[3]


def extract_sampled_bestfit(
    file: str | os.PathLike, sampled_only: bool = True
) -> dict[str, float]:
    file = Path(file)
    params: dict[str, float] = {}
    with file.open() as f:
        n = 0
        for line in f:
            if line.strip() == "":
                if n == 1 and sampled_only:
                    break
                n += 1
            elif n == 0:
                continue
            else:
                info = line.strip().split()
                params[info[2]] = float(info[1])
    return params


def satisfy_prerequisite(yaml_file: str | os.PathLike) -> bool:
    yaml_file = Path(yaml_file)
    if not yaml_file.name.endswith(".input.yaml"):
        print("ERROR: input file does not end with .input.yaml")
        return False
    minimum_file = yaml_file.parent / (
        yaml_file.name.replace(".input.yaml", ".minimum")
    )
    if not minimum_file.exists():
        print("ERROR: cannot find .minimum file")
        return False
    return True


def compute_bG_bestfit(model: Model, likelihoods: list[str]):
    ret: dict[str, float] = {}
    for l in likelihoods:
        try:
            # nomarg
            ret.update(model.likelihood[l].bG_bestfit())
        except AttributeError:
            pass
    return ret


def build_model(
    yaml_file, params: dict[str, float], requires: dict, base: Path | None = None
):
    base = base or Path.cwd()
    info = yaml_load_file(str(yaml_file))
    for name, value in params.items():
        if (v := info["params"].get(name, NOTFOUND)) is NOTFOUND:
            info["params"][name] = value
        elif isinstance(v, dict):
            info["params"][name]["value"] = value
            info["params"][name].pop("prior", None)
            info["params"][name].pop("ref", None)
        else:
            info["params"][name] = value
    for k, v in info["likelihood"].items():
        if v.get("marg"):
            v["marg"] = None
    info = yaml_dump(info)
    with PathContext(base):
        model = get_model(info)
        sampled_point = model.parameterization.sampled_params()
        for name in sampled_point:
            sampled_point[name] = params[name]
        model.add_requirements(requires)
    model.logpost(sampled_point)
    if names := [k for k in requires if k.endswith("_chi2")]:
        for name in names:
            ndata = model.likelihood[name.split("_chi2")[0]].ndata
            print(f"{name} = {model.provider.get_param(name):.2f}/{ndata}")
    return model


def generate_requires(
    tracers: list[str], hex: list[str], chained: list[str], likelihoods: list[str] = []
):
    common = {}
    if hex:
        common["ls"] = [0, 2, 4]
    else:
        common["ls"] = [0, 2]
    if chained:
        common["chained"] = True
    ret = {}
    for tracer in tracers:
        value = {}
        value["ls"] = [0, 2, 4] if tracer in hex else [0, 2]
        if tracer in chained:
            value["chained"] = True
        ret[tracer] = value
    ret = {"nonlinear_Plk_interpolator": ret}
    ret.update({k + "_chi2": None for k in likelihoods})  # type: ignore
    return ret


def get_cov(like: EFTLikeSingle | EFTLikeDouble | EFTLikeDoubleCross | EFTLike):
    cov = find_covariance_reader(
        like.cov.get("reader", "auto"),
        like.cov.get("reader_kwargs", {}),
    )(like.cov["path"])
    cov /= like.cov.get("rescale", 1)
    return cov


def build_multipole_dataframe(minfo: MultipoleInfo, err):
    df = minfo.df.copy(deep=True)
    for ell, Perr in zip(minfo.ls_tot, np.split(err, len(minfo.ls_tot))):
        df[minfo.symbol + f"{ell}err"] = Perr
    return df


def collect_multipole_dataframe(model, tracers: list[str], likelihoods: list[str]):
    configdict: dict[str, tuple[MultipoleInfo, Any]] = {}
    for like in (v for k, v in model.likelihood.items() if k in likelihoods):
        if isinstance(like, EFTLikeSingle):
            configdict[like.tracer] = (like.minfo, np.sqrt(get_cov(like).diagonal()))
        elif isinstance(like, (EFTLikeDouble, EFTLikeDoubleCross)):
            indices = itertools.accumulate(
                (len(x.ls_tot) * x.df.index.size for x in like.minfodict.values()),
                initial=0,
            )
            indices = list(indices)[1:-1]
            for k, err in zip(like.tracer, np.split(get_cov(like).diagonal(), indices)):
                configdict[k] = (like.minfodict[k], np.sqrt(err))
        elif isinstance(like, EFTLike):
            indices = itertools.accumulate(
                (len(x.ls_tot) * x.df.index.size for x in like.minfodict.values()),
                initial=0,
            )
            indices = list(indices)[1:-1]
            for k, err in zip(
                like.tracers, np.split(get_cov(like).diagonal(), indices)
            ):
                configdict[k] = (like.minfodict[k], np.sqrt(err))
        else:
            raise TypeError(f"Unexpected likelihood type: {type(like)}")
    configdict = {k: v for k, v in configdict.items() if k in tracers}
    return {
        tracer: build_multipole_dataframe(config[0], config[1])
        for tracer, config in configdict.items()
    }


def paint_data_and_theory(
    ax: Axes,
    df: pd.DataFrame,
    interpfn,
    hex: bool = False,
    chained: bool = False,
    ymin0: float | None = None,
    ymax0: float | None = None,
):
    P, _ = extract_multipole_info(df.columns.to_list())
    x = df.index.to_numpy()
    xx = np.geomspace(x.min(), x.max(), 1000)
    estyle = dict(capsize=2)  # elinewidth=0.75
    ymin, ymax = 0, 0
    if hex:
        ax.errorbar(
            x,
            x * df[f"{P}4"],
            yerr=x * df[f"{P}4err"],
            fmt=".",
            c="g",
            **estyle,
        )
        ymin = min(ymin, (x * (df[f"{P}4"] - 3 * df[f"{P}4err"])).min())
        ymax = max(ymax, (x * (df[f"{P}4"] + 3 * df[f"{P}4err"])).max())
        ax.plot(xx, xx * interpfn(4, xx), c="g")
    ax.errorbar(x, x * df[f"{P}2"], yerr=x * df[f"{P}2err"], fmt=".", c="b", **estyle)
    ymin = min(ymin, (x * (df[f"{P}2"] - 3 * df[f"{P}2err"])).min())
    ymax = max(ymax, (x * (df[f"{P}2"] + 3 * df[f"{P}2err"])).max())
    ax.plot(xx, xx * interpfn(2, xx), c="b")
    ax.errorbar(x, x * df[f"{P}0"], yerr=x * df[f"{P}0err"], fmt=".", c="k", **estyle)
    ymin = min(ymin, (x * (df[f"{P}0"] - 3 * df[f"{P}0err"])).min())
    ymax = max(ymax, (x * (df[f"{P}0"] + 3 * df[f"{P}0err"])).max())
    ax.plot(xx, xx * interpfn(0, xx), c="k")
    ax.set_xlabel(R"$k$ $[h\,\mathrm{Mpc}^{-1}]$")
    if chained:
        ax.set_ylabel(R"$kQ_\ell(k)$ $[h^{-1}\,\mathrm{Mpc}]^2$")
    else:
        ax.set_ylabel(R"$kP_\ell(k)$ $[h^{-1}\,\mathrm{Mpc}]^2$")
    ax.set_xlim(0, 0.3)
    if ymin0 is not None:
        ymin = ymin0
    if ymax0 is not None:
        ymax = ymax0
    ax.set_ylim(ymin, ymax)


def generate_data_like_theory(path: str, df: pd.DataFrame, interpfn: PlkInterpolator):
    if Path(path).exists():
        raise FileExistsError(f"File {path} already exists")
    symbol, ells = extract_multipole_info(df.columns.to_list())
    ells = [x for x in ells if x <= 4]
    k = df.index.to_numpy()
    try:
        out = interpfn(ells, k)
    except ValueError:
        ells = [x for x in ells if x <= 2]
        out = interpfn(ells, k)
    out = np.vstack((k, *out)).T
    header = (5 * " ").join(["k"] + [symbol + str(x) for x in ells])
    np.savetxt(path, out, header=header)


def get_argparser():
    parser = argparse.ArgumentParser(
        "mkbestfit",
        description="make best fit theory curve plot",
    )
    parser.add_argument("input_yaml", type=Path)
    parser.add_argument(
        "--likelihoods",
        nargs="+",
        required=True,
        metavar="LIKELIHOOD",
        help="likelihood list",
    )
    parser.add_argument(
        "--tracers", nargs="+", required=True, metavar="TRACER", help="tracer list"
    )
    parser.add_argument("-o", "--output", default="test.pdf", help="(default test.pdf)")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    parser.add_argument(
        "--freeze",
        nargs=2,
        action="append",
        metavar=("TRACER", "FILE"),
        default=[],
        help="freeze TRACER's theoretical output to FILE. "
        "Note, computed power spectrum does not include the binning effect",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path.cwd(),
        help="working path when loading yaml file (default cwd)",
    )
    parser.add_argument(
        "--hex",
        nargs="*",
        default=[],
        metavar="TRACER",
        help="compute and plot hexadecapole for specified tracers, 'all' for all tracers",
    )
    parser.add_argument(
        "--chained",
        nargs="*",
        default=[],
        metavar="TRACER",
        help="compute and plot chained power spectrum for specified tracers, 'all' for all tracers",
    )
    parser.add_argument(
        "--sharey", action="store_true", help="share ylim for all subplots"
    )
    parser.add_argument(
        "--ymin", type=float, help="set the minimum of y axis for all subplots"
    )
    parser.add_argument(
        "--ymax", type=float, help="set the maximum of y axis for all subplots"
    )
    return parser


def main(input_args: Sequence[str] | None = None, save: bool = False):
    parser = get_argparser()
    args = parser.parse_args(input_args)

    if args.hex and args.hex[0] == "all":
        args.hex = args.tracers
    if args.chained and args.chained[0] == "all":
        args.chained = args.tracers

    yaml_file: Path = args.input_yaml.resolve()
    base: Path = args.base.resolve()
    verbose_guard = do_nothing if args.verbose else disable_logging

    if not satisfy_prerequisite(yaml_file):
        parser.exit()
    bestfit = extract_sampled_bestfit(
        yaml_file.parent / yaml_file.name.replace(".input.yaml", ".minimum")
    )
    with PathContext(base):
        with verbose_guard():
            model = get_model(yaml_file)
    model.logpost(bestfit)
    bestfit.update(compute_bG_bestfit(model, args.likelihoods))
    print("bestfit:")
    pprint.pprint(bestfit)
    with PathContext(base):
        with verbose_guard():
            mdict = collect_multipole_dataframe(model, args.tracers, args.likelihoods)

    with verbose_guard():
        best_model = build_model(
            yaml_file,
            bestfit,
            requires=generate_requires(
                args.tracers, args.hex, args.chained, args.likelihoods
            ),
            base=base,
        )

    width = 5 * len(args.tracers)
    height = 4
    sharey = False
    if args.sharey or args.ymin is not None or args.ymax is not None:
        sharey = True
    fig, axes = plt.subplots(
        1, len(args.tracers), figsize=(width, height), sharey=args.sharey
    )
    if len(args.tracers) == 1:
        axes: Any = [axes]
    freeze: dict[str, str] = {x[0]: x[1] for x in args.freeze}
    mdict = {t: mdict[t] for t in args.tracers}
    for i, (tracer, df) in enumerate(mdict.items()):
        fn = best_model.provider.get_nonlinear_Plk_interpolator(
            tracer, chained=(True if tracer in args.chained else False)
        )
        if path := freeze.get(tracer):
            generate_data_like_theory(path, df, fn)
        paint_data_and_theory(
            axes[i],
            df,
            fn,
            hex=(True if tracer in args.hex else False),
            chained=(True if tracer in args.chained else False),
            ymin0=args.ymin,
            ymax0=args.ymax,
        )
        axes[i].set_title(tracer)
    fig.tight_layout()
    if save:
        plt.savefig(args.output)
    return fig, axes


if __name__ == "__main__":
    main(save=True)
