from __future__ import annotations
import argparse
import os
import pprint
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Any, Callable
from cobaya import get_model
from cobaya.yaml import yaml_dump
from cobaya.yaml import yaml_load_file
from eftpipe.lssdata import Multipoles
from eftpipe.tools import PathContext


def extract_sampled_bestfit(file: str | os.PathLike) -> dict[str, float]:
    file = Path(file)
    params: dict[str, float] = {}
    with file.open() as f:
        n = 0
        for line in f:
            if line.strip() == "":
                if n == 1:
                    break
                n += 1
            elif n == 0:
                continue
            else:
                info = line.strip().split()
                params[info[2]] = float(info[1])
    return params


def check_prerequisite(yaml_file: Path, exit: Callable[[], None]):
    if not yaml_file.name.endswith(".input.yaml"):
        print("ERROR: input file does not end with .input.yaml")
        exit()
    minimum_file = yaml_file.parent / (
        yaml_file.name.replace(".input.yaml", ".minimum")
    )
    if not minimum_file.exists():
        print("ERROR: cannot find .minimum file")
        exit()


def compute_bG_bestfit(model, likelihoods: list[str]):
    ret: dict[str, float] = {}
    for l in likelihoods:
        ret.update(model.likelihood[l].bG_bestfit())
    return ret


def get_bestfit_model(
    yaml_file, bestfit: dict[str, float], requires: dict, base: Path | None = None
):
    base = base or Path.cwd()
    info = yaml_load_file(str(yaml_file))
    for name, value in bestfit.items():
        if "value" in info["params"][name]:
            info["params"][name]["value"] = value
    info["likelihood"] = {"one": None}
    info = yaml_dump(info)
    with PathContext(base):
        model = get_model(info)
    sampled_point = model.parameterization.sampled_params()
    for name in sampled_point:
        sampled_point[name] = bestfit[name]
    model.add_requirements(requires)
    model.logpost(sampled_point)
    return model


def generate_requires(tracers: list[str], hex: bool, chained: bool):
    common = {}
    if hex:
        common["ls"] = [0, 2, 4]
    else:
        common["ls"] = [0, 2]
    if chained:
        common["chained"] = True
    ret = {"nonlinear_Plk_interpolator": {name: common for name in tracers}}
    return ret


def paint_data_and_theory(
    ax, m: Multipoles, interpfn, hex: bool = False, chained: bool = False
):
    if hex:
        ax.errorbar(m[4].x, m[4].x * m[4].y, yerr=m[4].x * m[4].yerr, fmt=".", c="g")
        ax.plot(m[4].x, m[4].x * interpfn(4, m[4].x), c="g")
    ax.errorbar(m[2].x, m[2].x * m[2].y, yerr=m[2].x * m[2].yerr, fmt=".", c="b")
    ax.plot(m[2].x, m[2].x * interpfn(2, m[2].x), c="b")
    ax.errorbar(m[0].x, m[0].x * m[0].y, yerr=m[0].x * m[0].yerr, fmt=".", c="k")
    ax.plot(m[0].x, m[0].x * interpfn(0, m[0].x), c="k")
    ax.set_xlabel(r"$k$ $[h\,\mathrm{Mpc}^{-1}]$")
    if chained:
        ax.set_ylabel(r"$kQ_\ell(k)$ $[h^{-1}\,\mathrm{Mpc}]^2$")
    else:
        ax.set_ylabel(r"$kP_\ell(k)$ $[h^{-1}\,\mathrm{Mpc}]^2$")


def generate_data_like_theory(path, m: Multipoles, interpfn):
    k = m.x_all
    tmp = [k]
    label = ["k"]
    for ell in m.raw.keys():
        tmp.append(interpfn(ell, k))
        label.append(m.yname + str(ell))
    out = np.vstack(tmp).T
    np.savetxt(path, out, header=(5 * " ").join(label))


def main():
    parser = argparse.ArgumentParser(
        "mkbestfit", description="make best fit theory curve plot"
    )
    parser.add_argument("yaml", type=Path)
    parser.add_argument("--likelihoods", nargs="+", required=True)
    parser.add_argument("--tracers", nargs="+", required=True)
    parser.add_argument("-o", default="test.pdf")
    parser.add_argument(
        "-th", nargs="+", default=None, help="generate theoretical 'data'"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path.cwd(),
        help="working path when loading yaml file, by default cwd",
    )
    parser.add_argument("--hex", action="store_true")
    parser.add_argument("--chained", action="store_true")
    args = parser.parse_args()

    yaml_file: Path = args.yaml
    check_prerequisite(yaml_file, parser.exit)
    bestfit = extract_sampled_bestfit(
        yaml_file.parent / yaml_file.name.replace(".input.yaml", ".minimum")
    )
    with PathContext(args.base):
        model = get_model(yaml_file)
    model.logpost(bestfit)
    bestfit.update(compute_bG_bestfit(model, args.likelihoods))
    print("bestfit:")
    pprint.pprint(bestfit)

    best_model = get_bestfit_model(
        yaml_file,
        bestfit,
        requires=generate_requires(args.tracers, args.hex, args.chained),
        base=args.base,
    )

    width = 5 * len(args.tracers)
    height = 4
    fig, axes = plt.subplots(1, len(args.tracers), figsize=(width, height))
    if len(args.tracers) == 1:
        axes: Any = [axes]
    for tracer, like, i in zip(
        args.tracers, args.likelihoods, range(len(args.tracers)),
    ):
        fn = best_model.provider.get_nonlinear_Plk_interpolator(
            tracer, chained=args.chained
        )
        m = model.likelihood[like].lssdata.fullshape[0]
        if args.th:
            generate_data_like_theory(args.th[i], m, fn)
        paint_data_and_theory(axes[i], m, fn, hex=args.hex, chained=args.chained)
        axes[i].set_title(tracer)
    fig.tight_layout()
    plt.savefig(args.o)


if __name__ == "__main__":
    main()
