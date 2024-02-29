"""
anything related to analysis
"""

from __future__ import annotations
import itertools
import logging
import math
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    cast,
    Callable,
    Dict,
    Iterable,
    Literal,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    TYPE_CHECKING,
    Union,
)
from cobaya import get_model
from cobaya.yaml import yaml_dump
from cobaya.yaml import yaml_dump_file
from cobaya.yaml import yaml_load_file
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from .reader import read_pkl
from .theory import PlkInterpolator
from .tools import pairwise, verbose_guard

if TYPE_CHECKING:
    from .parambasis import BirdComponent
    from .typing import FilePath, ndarrayf

MultipoleStyle = Dict[Union[int, Literal["default"]], "MultipoleStyleElement"]
_T = TypeVar("_T")


@dataclass
class MultipoleStyleElement:
    line: dict[str, Any]
    errorbar: dict[str, Any]


CLASSICAL_STYLE: MultipoleStyle = {
    0: MultipoleStyleElement(line={"c": "k"}, errorbar={"c": "k"}),
    2: MultipoleStyleElement(line={"c": "b"}, errorbar={"c": "b"}),
    4: MultipoleStyleElement(line={"c": "g"}, errorbar={"c": "g"}),
    "default": MultipoleStyleElement(line={"fmt": "-"}, errorbar={"fmt": ".", "capsize": 2}),
}

MODERN_STYLE: MultipoleStyle = {
    0: MultipoleStyleElement(
        line=dict(c="#993732"),
        errorbar=dict(
            fmt="o",
            c="#993732",
            mfc="#f1a83b",
            mec="#993732",
            ecolor="#993732",
        ),
    ),
    2: MultipoleStyleElement(
        line=dict(c="#5580b0"),
        errorbar=dict(
            fmt="D",
            c="#5580b0",
            mfc="#97ccf6",
            mec="#5580b0",
            ecolor="#5580b0",
        ),
    ),
    4: MultipoleStyleElement(
        line=dict(c="#818181"),
        errorbar=dict(
            fmt="s",
            c="#818181",
            mfc="#c0c0c0",
            mec="#818181",
            ecolor="#818181",
        ),
    ),
    "default": MultipoleStyleElement(
        line=dict(alpha=0.7),
        errorbar=dict(capsize=1.3, markersize=4),
    ),
}


def update_style(style: MultipoleStyle, default: dict = {}):
    style = deepcopy(style)
    if not style.get("default"):
        style["default"] = MultipoleStyleElement(line={}, errorbar={})
    style["default"].line.update(deepcopy(default))
    style["default"].errorbar.update(deepcopy(default))
    for ell, element in style.items():
        if ell == "default":
            continue
        element.line.update(deepcopy(style["default"].line))
        element.errorbar.update(deepcopy(style["default"].errorbar))
    return style


@dataclass
class CobayaProducts:
    root: str
    root_dir: Path

    chains: pd.DataFrame = field(init=False, repr=False, compare=False)
    input_info: dict[str, Any] = field(init=False, repr=False, compare=False)
    updated_info: dict[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        with open(self.root_dir / (self.root + ".1.txt")) as f:
            names = f.readline().strip().split()[1:]
        chains = []
        for i in range(1, self.nchains() + 1):
            chains.append(np.loadtxt(self.root_dir / f"{self.root}.{i}.txt"))
        self.chains = pd.DataFrame(np.vstack(chains), columns=names)
        self.input_info = yaml_load_file(str(self.input_yaml_file()))
        self.updated_info = yaml_load_file(str(self.updated_yaml_file()))

    @classmethod
    def from_yaml_file(cls, yaml_file: str):
        if yaml_file.endswith((".input.yaml", ".updated.yaml")):
            parent = Path(yaml_file).expanduser().resolve().parent
        else:
            parent = Path.cwd()
        info = yaml_load_file(yaml_file)
        output: Path = parent / info["output"]
        root_dir = output.parent
        root = output.name
        return cls(root=root, root_dir=root_dir)

    def nchains(self) -> int:
        pattern = re.compile(r".*\.(\d+)\.txt")
        n = 0
        for path in self.root_dir.glob(f"{self.root}.*.txt"):
            if pattern.match(path.name):
                n += 1
        return n

    def sampled_params(self) -> dict[str, dict]:
        retval: dict[str, dict] = {}
        for param, config in self.updated_info["params"].items():
            if "prior" in config:
                tmp = deepcopy(config)
                tmp.pop("renames", None)
                retval[param] = tmp
        return retval

    def fixed_params(self) -> dict[str, dict]:
        retval: dict[str, dict] = {}
        for param, config in self.updated_info["params"].items():
            if "value" in config and not config.get("derived", False):
                tmp = deepcopy(config)
                tmp.pop("renames", None)
                retval[param] = tmp
        return retval

    def fixed_params_values_dict(self) -> dict[str, float]:
        info = self.fixed_params()
        # XXX: it's possible that v["value"] is string
        return {k: v["value"] for k, v in info.items()}

    def derived_params(self) -> dict[str, dict]:
        retval: dict[str, dict] = {}
        for param, config in self.updated_info["params"].items():
            if config.get("derived", False):
                tmp = deepcopy(config)
                tmp.pop("renames", None)
                retval[param] = tmp
        return retval

    def all_params(self) -> dict[str, dict]:
        return {**self.sampled_params(), **self.fixed_params(), **self.derived_params()}

    def iloc(
        self,
        i: int,
        include_fixed: bool = True,
        sampled_only: bool = False,
        params_only: bool = False,
    ) -> dict[str, float]:
        params_values_dict = self.chains.iloc[i].to_dict()
        if include_fixed:
            params_values_dict.update(self.fixed_params_values_dict())
        if params_only:
            params_values_dict = {k: params_values_dict[k] for k in self.all_params()}
        if sampled_only:
            params_values_dict = {k: params_values_dict[k] for k in self.sampled_params()}
        return params_values_dict

    def bestfit(
        self,
        include_fixed: bool = True,
        sampled_only: bool = False,
        params_only: bool = False,
    ) -> dict[str, float]:
        # if .minimum file exists, use it
        if (file := self.minimum_file()).exists():
            params_values_dict: dict[str, float] = {}
            with file.open("r") as f:
                skiprows = 3  # skip -log(Like) and chi-sq
                for i, line in enumerate(f):
                    if i < skiprows:
                        continue
                    if not line.strip():
                        if sampled_only:
                            break
                        continue
                    idx, value, name, *latex = line.strip().split()
                    params_values_dict[name] = float(value)
            return params_values_dict
        # otherwise, find best fit from chains
        idx = cast(int, self.chains["minuslogpost"].idxmin())
        return self.iloc(idx, include_fixed, sampled_only, params_only)

    def root_string(self) -> str:
        return str(self.root_dir / self.root)

    def input_yaml_file(self) -> Path:
        return self.root_dir / f"{self.root}.input.yaml"

    def updated_yaml_file(self) -> Path:
        return self.root_dir / f"{self.root}.updated.yaml"

    def minimum_file(self) -> Path:
        return self.root_dir / f"{self.root}.minimum"

    def progress_file(self) -> Path:
        return self.root_dir / f"{self.root}.progress"

    def plot_progress(
        self,
        figure_kwargs: dict[str, Any] = {"figsize": (6, 4)},
        legend_kwargs: dict[str, Any] = {"frameon": False},
    ):
        from cobaya.samplers.mcmc import plot_progress

        plot_progress(
            self.root_string(),
            figure_kwargs=figure_kwargs,
            legend_kwargs=legend_kwargs,
        )
        plt.tight_layout()
        plt.show()

    def samples(self, ignore_rows: float = 0, no_cache: bool = False):
        import getdist

        return getdist.loadMCSamples(
            self.root_string(),
            no_cache=no_cache,
            settings={"ignore_rows": ignore_rows},
        )


@dataclass
class EFTLikeProducts(CobayaProducts):
    def marg_param_prefix(self) -> str:
        # XXX: hard-coded
        return "marg_"

    def fullchi2_suffix(self) -> str:
        # XXX: hard-coded
        return "_fullchi2"

    def marginalized_params(self, with_marg_prefix: bool = False) -> dict[str, dict[str, Any]]:
        from .likelihood import regularize_prior

        # step 1: collect marginalized params config
        marg: dict[str, dict[str, Any]] = {}
        for likename, config in self.input_info["likelihood"].items():
            if not supported_likelihood(likename, config):
                continue
            if margconfig := config.get("marg", None):
                marg.update(regularize_prior(margconfig))
        # step 2: update marg and keep necessary info only
        for p, config in marg.items():
            marg[p] = {
                "prior": {
                    "dist": "norm",
                    "loc": config.get("loc", 0),
                    "scale": config["scale"],
                },
                "ref": config.get("loc", 0),
                "proposal": config.get("proposal", 0.01),
                "latex": config.get("latex", p.replace("_", " ")),
            }
        if with_marg_prefix:
            prefix = self.marg_param_prefix()
            return {prefix + p: config for p, config in marg.items()}
        return marg

    def marginalized_prior(self, with_marg_prefix: bool = True):
        marg = self.marginalized_params(with_marg_prefix)

        def prior_function(params_values_dict: dict[str, Any]):
            chi2 = 0.0
            for p, config in marg.items():
                loc = config["prior"]["loc"]
                scale = config["prior"]["scale"]
                chi2 += (params_values_dict[p] - loc) ** 2 / scale**2
            return chi2

        return prior_function

    def cosmo_info(self):
        info = deepcopy(self.input_info)
        eftinfo = info["theory"].pop("eftpipe.eftlss", None)
        toremove_prefix = []
        if eftinfo:
            for t, v in eftinfo["tracers"].items():
                if t == "default":
                    continue
                toremove_prefix.append(v["prefix"])
        toremove_prefix = toremove_prefix + ["marg_" + _ for _ in toremove_prefix]
        info["likelihood"] = {"one": None}
        info.pop("output", None)
        info.pop("sampler", None)
        params = {
            p: v
            for p, v in info["params"].items()
            if (not any(p.startswith(_) for _ in toremove_prefix))
            and not p.endswith(self.fullchi2_suffix())
        }
        info["params"] = params
        return info

    def cosmo_model(self, debug: bool | None = None):
        return get_model(self.cosmo_info(), debug=debug)  # type: ignore

    def full_model_info(self) -> dict[str, Any]:
        # XXX: may be better to use updated_info
        info = deepcopy(self.input_info)
        # remove marg block
        for likename, config in info["likelihood"].items():
            if not supported_likelihood(likename, config):
                continue
            config.pop("marg", None)
            config.pop("jeffreys", None)
        # update info["params"] and remove derived params
        marg = self.marginalized_params()
        params = info["params"]
        for p, config in marg.items():
            if prior := config.get("prior"):
                if prior.get("scale") is None:
                    prior["scale"] = 1000
            params[p] = config
        marg_param_prefix = self.marg_param_prefix()
        for p in marg.keys():
            params.pop(marg_param_prefix + p, None)
        return info

    def dump_full_model_info(self, output: str | None = None, stream=None):
        info = self.full_model_info()
        if output:
            info["output"] = output
        return yaml_dump(info, stream)

    def dump_full_model_yaml(
        self,
        file_name: str,
        output: str | None = None,
        comment=None,
        error_if_exists=True,
    ):
        info = self.full_model_info()
        if output:
            info["output"] = output
        return yaml_dump_file(
            file_name,
            info,
            comment=comment,
            error_if_exists=error_if_exists,
        )

    def global_bestfit(
        self,
        include_fixed: bool = True,
        sampled_only: bool = False,
        params_only: bool = False,
        include_prior: bool = False,
    ):
        """
        Note
        ----
        ``include_prior``=True is more consistent: assume bNG is best fit and then
        compute the corresponding bG and -2logp; ``include_prior``=False does not
        depend on prior, but is similar to randomly choose a point and compute chi2.
        Anyway, both of them are approximate estimation of global best fit.
        """
        if self.using_jeffreys_prior():
            return self.bestfit(
                include_fixed=include_fixed,
                sampled_only=sampled_only,
                params_only=params_only,
            )
        fullchi2_names: list[str] = [
            k for k in self.chains.columns if k.endswith(self.fullchi2_suffix())
        ]
        # best fit from full chi2
        fullchi2 = sum(self.chains[name] for name in fullchi2_names)
        if include_prior:
            prior_function = self.marginalized_prior()
            fullchi2 += prior_function(self.chains.to_dict("series"))  # type: ignore
        idx: int = fullchi2.idxmin()  # type: ignore
        return self.iloc(idx, include_fixed, sampled_only, params_only)

    def global_bestfit_if_possible(
        self,
        include_fixed: bool = True,
        sampled_only: bool = False,
        params_only: bool = False,
        include_prior: bool = False,
        error_if_not_possible: bool = False,
    ) -> dict[str, float]:
        fullchi2_names: list[str] = [
            k for k in self.chains.columns if k.endswith(self.fullchi2_suffix())
        ]
        if not fullchi2_names:
            if error_if_not_possible:
                raise ValueError(
                    f"no {self.fullchi2_suffix()} computed, cannot find global best fit"
                )
            # best fit from (possibly-)marginalized posterior
            return self.bestfit(include_fixed, sampled_only, params_only)
        return self.global_bestfit(
            include_fixed, sampled_only, params_only, include_prior=include_prior
        )

    def using_jeffreys_prior(self) -> bool:
        """return True if all likelihoods are using Jeffreys prior"""
        for likename, config in self.input_info["likelihood"].items():
            if not supported_likelihood(likename, config):
                return False
            if not config.get("jeffreys", False):
                return False
        return True


@dataclass
class Multipole(Mapping[str, pd.Series]):
    ells: tuple[int, ...]
    x: ndarrayf = field(repr=False)
    data: pd.DataFrame = field(repr=False, compare=False)
    symbol: str = "P"
    style: MultipoleStyle = field(default_factory=lambda: deepcopy(MODERN_STYLE))

    @classmethod
    def init(cls, style: MultipoleStyle | None = None, **kwargs: ndarrayf):
        if len(kwargs) < 2:
            raise ValueError("at least two kwargs are required")
        xname = next(iter(kwargs))
        x = kwargs[xname]
        symbol, ells = cls.infer_symbol_and_ells(kwargs)
        data = pd.DataFrame(kwargs)
        data.set_index(xname, drop=False, inplace=True)
        return cls(ells, x, data, symbol, style=style or deepcopy(MODERN_STYLE))

    @classmethod
    def from_dataframe(cls, df):
        symbol, ells = cls.infer_symbol_and_ells(df.columns)
        return cls(ells, df.index.to_numpy(), df)

    @classmethod
    def loadtxt(
        cls,
        path: FilePath,
        cov_path: FilePath | None = None,
        logger: logging.Logger | None = None,
        style: MultipoleStyle | None = None,
    ):
        """load multipole from txt file, assume power spectrum by default"""
        data = read_pkl(path, logger=logger)
        x = data.index.to_numpy()
        symbol, ells = cls.infer_symbol_and_ells(data.columns)
        multipole = cls(ells, x, data, symbol, style=style or deepcopy(MODERN_STYLE))
        if cov_path:
            multipole.apply_covariance(np.loadtxt(cov_path))
        return multipole

    @classmethod
    def infer_symbol_and_ells(cls, names: Iterable[str]):
        pattern = re.compile(r"(?P<symbol>[A-Za-z]+)(?P<ell>\d+)")
        db = defaultdict(list)
        for name in names:
            if match := pattern.match(name):
                db[match.group("symbol")].append(int(match.group("ell")))
        cnt = Counter({k: len(v) for k, v in db.items()})
        symbol = cast(str, cnt.most_common(1)[0][0])
        ells = cast(Tuple[int, ...], tuple(sorted(db[symbol])))
        return symbol, ells

    @property
    def k(self):
        return self.x

    @property
    def s(self):
        return self.x

    def mono(self) -> pd.Series:
        return self.data[self.symbol + "0"]

    def mono_err(self) -> pd.Series | None:
        return self.get(self.symbol + "0err")

    def quad(self) -> pd.Series:
        return self.data[self.symbol + "2"]

    def quad_err(self) -> pd.Series | None:
        return self.get(self.symbol + "2err")

    def hex(self) -> pd.Series:
        return self.data[self.symbol + "4"]

    def hex_err(self) -> pd.Series | None:
        return self.get(self.symbol + "4err")

    def data_vector(self) -> ndarrayf:
        return np.hstack(list(self.values()))

    def clone(self):
        return self.__class__(
            ells=self.ells,
            x=self.x.copy(),
            data=self.data.copy(),
            symbol=self.symbol,
            style=deepcopy(self.style),
        )

    def savetxt(
        self,
        fname: FilePath,
        fmt: str | Callable[[Any], str] | None = "%.18e",
        extra_info: str = "",
    ):
        with open(fname, "w") as f:
            column_names = self.data.columns.to_list()
            if self.data.index.name not in column_names:
                column_names = [self.data.index.name] + column_names
            header = (" " * 5).join(column_names)
            f.write(f"# {header}\n")
            if extra_info:
                f.write("# " + extra_info.replace("\n", "\n# ") + "\n")
            self.data.to_csv(f, sep=" ", header=False, index=True, float_format=fmt)

    def maybe_power_spectrum(self) -> bool:
        if self.symbol == "P":
            return True
        if self.x.max() < 10:
            return True
        return False

    def apply_covariance(self, cov: ndarrayf):
        sections = list(
            itertools.accumulate(self.values(), lambda total, elem: total + elem.size, initial=0)
        )[1:-1]
        errs = np.split(np.sqrt(cov.diagonal()), sections)
        for ell, yerr in zip(self.ells, errs):
            self.data[f"{self.symbol}{ell}err"] = yerr
        return self

    # plot
    def _plot(self, ax, ell, x, y, yerr, style: MultipoleStyle, label: str | None = None):
        extra = {}
        if label and ell == min(self.ells):
            extra["label"] = label
        if yerr is None:
            ax.plot(x, y, **style.get(ell, style["default"]).line, **extra)
        else:
            ax.errorbar(x, y, yerr=yerr, **style.get(ell, style["default"]).errorbar, **extra)

    def plot_pk(self, ax=None, label: str | None = None, compact: bool = True, **style):
        style = update_style(deepcopy(self.style), style)
        if ax is None:
            ax = plt.gca()
        k = self.k

        def y_and_yerr(Pk, Pkerr):
            y = k**1.5 * Pk if compact else k * Pk
            if Pkerr is None:
                return y, None
            yerr = k**1.5 * Pkerr if compact else k * Pkerr
            return y, yerr

        if (Pk := self.get(self.symbol + "4")) is not None:
            y, yerr = y_and_yerr(Pk, self.hex_err())
            self._plot(ax, 4, k, y, yerr, style, label)
        if (Pk := self.get(self.symbol + "2")) is not None:
            y, yerr = y_and_yerr(Pk, self.quad_err())
            self._plot(ax, 2, k, y, yerr, style, label)
        if (Pk := self.get(self.symbol + "0")) is not None:
            y, yerr = y_and_yerr(Pk, self.mono_err())
            self._plot(ax, 0, k, y, yerr, style, label)
        return ax

    def plot_xi(self, ax=None, label: str | None = None, **style):
        style = update_style(deepcopy(self.style), style)
        if ax is None:
            ax = plt.gca()
        s = self.s
        if (xi := self.get(self.symbol + "4")) is not None:
            xierr = self.hex_err()
            xierr = None if xierr is None else s**2 * xierr
            self._plot(ax, 4, s, s**2 * xi, xierr, style, label)
        if (xi := self.get(self.symbol + "2")) is not None:
            xierr = self.quad_err()
            xierr = None if xierr is None else s**2 * xierr
            self._plot(ax, 2, s, s**2 * xi, xierr, style, label)
        if (xi := self.get(self.symbol + "0")) is not None:
            xierr = self.mono_err()
            xierr = None if xierr is None else s**2 * xierr
            self._plot(ax, 0, s, s**2 * xi, xierr, style, label)
        return ax

    def plot(self, ax=None, label: str | None = None, compact: bool = False, **style):
        if self.maybe_power_spectrum():
            ax = self.plot_pk(ax, label, compact=compact, **style)
            ax.set_xlabel(R"$k$ $[h\,\mathrm{Mpc}^{-1}]$")
            if not compact:
                ax.set_ylabel(Rf"$k{self.symbol}_\ell(k)$ $[h^{{-1}}\,\mathrm{{Mpc}}]^2$")
            else:
                ax.set_ylabel(
                    Rf"$k^{{3/2}}{self.symbol}_\ell(k)$ $[h^{{-1}}\,\mathrm{{Mpc}}]^{{3/2}}$"
                )
        else:
            ax = self.plot_xi(ax, label, **style)
            ax.set_xlabel(R"$s$ $[h^{-1}\,\mathrm{Mpc}]$")
            ax.set_ylabel(Rf"$s^2{self.symbol}_\ell(s)$ $[h^{{-1}}\,\mathrm{{Mpc}}]^2$")
        return ax

    # Mapping interface
    def get(self, key: str, default: _T = None) -> pd.Series | _T:
        return self.data.get(key, default)  # type: ignore

    def __getitem__(self, key: str) -> pd.Series:
        return self.data[key]

    def __iter__(self):
        return iter(self.symbol + str(ell) for ell in self.ells)

    def __len__(self) -> int:
        return len(self.ells)

    def __getattr__(self, name: str) -> pd.Series:
        if name in self.data.columns:
            return self.data[name]
        raise AttributeError(f"no such attribute: {name}")


def supported_likelihood(likename: str, config: dict[str, Any]) -> bool:
    supported = ["eftpipe.eftlike", "eftpipe.likelihood.EFTLike"]
    return likename in supported or config.get("class") in supported


def collect_multipoles(info: dict[str, dict]) -> dict[str, Multipole]:
    from .likelihood import EFTLike, find_data_reader

    tracer_multipoles: dict[str, Multipole] = {}
    for likename, config in info["likelihood"].items():
        if not supported_likelihood(likename, config):
            continue
        # initialize the likelihood, which regularizes attributes
        likelihood = EFTLike(config)
        full_covmat = likelihood.full_covmat
        tracers = likelihood.tracers

        nsize = []
        for tracer in tracers:
            _d = likelihood.data[tracer]
            reader = find_data_reader(_d.get("reader"), **_d.get("reader_kwargs", {}))
            df = reader(_d["path"])
            multipole = Multipole.from_dataframe(df)
            tracer_multipoles[tracer] = multipole
            nsize.append(multipole.data_vector().size)
        for (istart, iend), tracer in zip(
            pairwise(itertools.accumulate(nsize, initial=0)), tracers
        ):
            subcov = full_covmat[istart:iend, istart:iend]
            tracer_multipoles[tracer].apply_covariance(subcov)
    return tracer_multipoles


def paint_multipole(
    ells: list[int],
    k: ndarrayf,
    Plk: PlkInterpolator,
    ax=None,
    label: str | None = None,
    compact: bool = False,
    style: MultipoleStyle = MODERN_STYLE,
):
    if ax is None:
        ax = plt.gca()
    extra = {} if label is None else {"label": label}
    if 0 in ells:
        y = k**1.5 * Plk(0, k) if compact else k * Plk(0, k)
        ax.plot(k, y, **style[0].line, **extra)
    if 2 in ells:
        y = k**1.5 * Plk(2, k) if compact else k * Plk(2, k)
        ax.plot(k, y, **style[2].line)
    if 4 in ells:
        y = k**1.5 * Plk(4, k) if compact else k * Plk(4, k)
        ax.plot(k, y, **style[4].line)
    return ax


@dataclass
class BestfitModel:
    yaml_file: str
    verbose: bool = False
    component: bool = False
    # evaluate the bestfit model without window
    remove_window: bool = False
    # evaluate the bestfit model at different zeff
    zeff: float | None = None

    def __post_init__(self):
        from .likelihood import EFTLike

        # step 1: get global bestfit
        products = EFTLikeProducts.from_yaml_file(self.yaml_file)
        marg_param_prefix = products.marg_param_prefix()
        itrim = len(marg_param_prefix)
        bestfit = products.global_bestfit_if_possible()
        bestfit = {
            (k[itrim:] if k.startswith(marg_param_prefix) else k): v for k, v in bestfit.items()
        }
        # step 2: get requirements, tracers and chained
        info = yaml_load_file(self.yaml_file)
        requires: defaultdict[str, Any] = defaultdict(dict)
        tracers: list[str] = []
        chained: dict[str, bool] = {}
        fullchi2: list[str] = []
        ndata: list[int] = []
        hartlap: list[float] = []
        for likename, config in info["likelihood"].items():
            if not supported_likelihood(likename, config):
                continue
            with verbose_guard(False):
                likelihood = EFTLike(config)
            tracers.extend(likelihood.tracers)
            chained.update(likelihood.chained)
            for tracer, chained_, ells in zip(
                likelihood.tracers,
                likelihood.chained.values(),
                (_.ls for _ in likelihood.minfodict.values()),
            ):
                requires["nonlinear_Plk_interpolator"][tracer] = {
                    "ls": ells,
                    "chained": chained_,
                }
                if self.component:
                    requires["bird_component"][tracer] = None
            fullchi2.append(likename + "_fullchi2")
            ndata.append(likelihood.ndata)
            hartlap.append(likelihood.hartlap if likelihood.hartlap is not None else 1)
            requires[likename + "_fullchi2"] = None
        # step 3: evaluate full model
        fullinfo = products.full_model_info()
        if self.remove_window:
            fullinfo["theory"]["eftpipe.eftlss"]["tracers"]["default"]["with_window"] = False
        if self.zeff is not None:
            _tracers_info = fullinfo["theory"]["eftpipe.eftlss"]["tracers"]
            for x in _tracers_info.values():
                x["z"] = self.zeff
        with verbose_guard(self.verbose):
            model = get_model(fullinfo)  # type: ignore
            model.add_requirements(requires)
        point = model.parameterization.sampled_params()
        try:
            point = {k: bestfit[k] for k in point.keys()}
        except KeyError:
            # missing the bestfit of gaussian parameters
            with verbose_guard(False):
                m = get_model(info)  # type: ignore
                p = m.parameterization.sampled_params()
                p = {k: bestfit[k] for k in p.keys()}
                m.logpost(p)
                for like in m.likelihood.values():
                    if isinstance(like, EFTLike):
                        bestfit.update(like.bG_bestfit())
            point = {k: bestfit[k] for k in point.keys()}
        model.logpost(point)

        self.tracers = tracers
        self.model = model
        self.bestfit = bestfit
        self.products = products
        with verbose_guard(False):
            self.multipoles = collect_multipoles(info)
        self.chained = chained
        # chi2 w/o hartlap correction
        self.fullchi2 = {
            k: f"{self.model.provider.get_param(k) / h:.3f}/{n}"
            for k, h, n in zip(fullchi2, hartlap, ndata)
        }
        self.fullchi2_hartlap = {
            k: f"{self.model.provider.get_param(k):.3f}/{n}" for k, n in zip(fullchi2, ndata)
        }

    def Plk_interpolator(self, tracer: str) -> PlkInterpolator:
        return self.model.provider.get_nonlinear_Plk_interpolator(
            tracer, chained=self.chained[tracer]
        )

    def bird_component(self, tracer: str) -> tuple[list[int], ndarrayf, BirdComponent]:
        return self.model.provider.get_bird_component(tracer)

    def plot(self, tracer: str, ax=None, compact: bool = False, **errorbar_style):
        if ax is None:
            ax = plt.gca()
        self.multipoles[tracer].plot(ax, compact=compact, **errorbar_style)
        k = np.linspace(0.0005, 0.3, 1000)
        Plk = self.Plk_interpolator(tracer)
        _style = update_style(self.multipoles[tracer].style, errorbar_style)
        paint_multipole(Plk.ls, k, Plk, ax=ax, compact=compact, style=_style)
        # ax.set_title(tracer.replace("_", " ").replace("X", "Cross"))
        text = tracer.replace("_", " ").replace("X", "Cross")
        ax.text(
            0.94,
            0.90,
            text,
            fontdict={"fontsize": 12},
            transform=ax.transAxes,
            ha="right",
            va="center",
        )
        # ax.minorticks_on()
        ax.tick_params(which="both", direction="in")
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        # ax.grid(c="c", ls="--", lw=1)
        return ax

    def plot_theory(self, tracer: str, ax=None, compact: bool = False, **errorbar_style):
        if ax is None:
            ax = plt.gca()
        k = np.linspace(0.0005, 0.3, 1000)
        Plk = self.Plk_interpolator(tracer)
        _style = update_style(self.multipoles[tracer].style, errorbar_style)
        paint_multipole(Plk.ls, k, Plk, ax=ax, compact=compact, style=_style)
        return ax

    def plot_component(self, tracer: str, ax=None):
        if ax is None:
            ax = plt.gca()
        ls, kraw, component = self.bird_component(tracer)
        Plin, Ploop, Pct = (
            PlkInterpolator(ls, kraw, np.abs(p))
            for p in (component.Plin, component.Ploop, component.Pct)
        )

        k = np.linspace(0.0005, 0.3, 1000)
        # TODO: ls="-", "--" and ":"
        paint_multipole(Plin.ls, k, Plin, ax=ax, label="linear")
        paint_multipole(Ploop.ls, k, Ploop, ax=ax, label="loop")
        paint_multipole(Pct.ls, k, Pct, ax=ax, label="counter")
        ax.legend(frameon=False)
        ax.set_xlabel(R"$k$ $[h\,\mathrm{Mpc}^{-1}]$")
        ax.set_ylabel(R"$|kP_\ell(k)|$ $[h^{-1}\,\mathrm{Mpc}]^2$")
        ax.set_title(tracer.replace("_", " "))
        return ax


def efunc(z: float, omegam: float) -> float:
    """H(z) / H0"""
    try:
        return math.sqrt(omegam * (1 + z) ** 3 + 1 - omegam)
    except ValueError:
        return math.nan


def DH(z: float, omegam: float) -> float:
    """D_H(z) in hubble unit, Mpc/h"""
    return 299792.458 / (100 * efunc(z, omegam))


def DM(z: float, omegam: float) -> float:
    """D_M(z) in hubble unit, Mpc/h"""
    return quad(DH, 0, z, args=(omegam,))[0]


def omegam_z(z: Any, omegam: Any) -> Any:
    """Omega_m(z)"""
    return omegam * (1 + z) ** 3 / (omegam * (1 + z) ** 3 + 1 - omegam)


def growth_factor(z: float, omegam: float, normalize: bool = True) -> float:
    """linear growth factor

    References
    ----------
    Dodelson, Modern Cosmology, Eq. 8.77
    """
    # fmt: off
    gz = (
        5/2 * omegam * efunc(z, omegam)
        * quad(lambda t: (1 + t) / efunc(t, omegam) ** 3, z, np.inf)[0]
    )
    norm = 1
    if normalize:
        norm = (
            5/2 * omegam * efunc(0, omegam)
            * quad(lambda t: (1 + t) / efunc(t, omegam) ** 3, 0, np.inf)[0]
        )
    # fmt: on
    return gz / norm


def growth_rate(z: float, omegam: float, fast: bool = False) -> float:
    if fast:
        return omegam_z(z, omegam) ** 0.55
    # fmt: off
    return (
        3 * omegam / (2 * (omegam - 1) / (1 + z)**3 - 2 * omegam)
        + 5/2 * omegam * (1 + z)**2 / efunc(z, omegam)**2 / growth_factor(z, omegam, normalize=False)
    )
    # fmt: on


def cosmo_to_fsigma8(z, cosmo_model, chains: pd.DataFrame, output: str | None = None):
    assert isinstance(chains.index, pd.RangeIndex)
    assert chains.index.start == 0

    def format_array(arr):
        fn = lambda x: f"{x:.16f}"
        return " ".join(map(fn, arr))

    if not isinstance(z, list):
        z = [z]
    cosmo_model.add_requirements({"fsigma8": {"z": z}})
    fsigma8 = []
    point = cosmo_model.parameterization.sampled_params()
    with open(output if output else os.devnull, "w") as f:
        if output:
            f.write(f"# {' '.join(map(str, z))}\n")
        for i in range(chains.shape[0]):
            for k in point.keys():
                point[k] = chains.loc[i, k]  # type: ignore
            cosmo_model.logpost(point)
            _fsigma8 = cosmo_model.provider.get_fsigma8(z)
            fsigma8.append(_fsigma8)
            if output:
                f.write(format_array(_fsigma8) + "\n")
    return np.array(fsigma8)


@dataclass(frozen=True)
class LssConvertor:
    """convert lss analysis results like fsigma8, alperp and alpara into suitable format

    Parameters
    ----------
    alpara: float
        BAO parameter along the line of sight
    alperp: float
        BAO parameter perpendicular to the line of sight
    fsigma8: float
        f * sigma_8
    zeff: float
        effective redshift
    omegam_fid: float
        fiducial Omega_m
    rdrag_fid: float
        fiducial r_drag, in Mpc unit
    H0_fid: float
        fiducial H0, in km/s/Mpc unit

    Notes
    -----
    This class aims to provide a user-friendly interface to convert lss analysis results

    References
    ----------
    -  https://arxiv.org/abs/2007.08991
    """

    alpara: float
    alperp: float
    fsigma8: float
    zeff: float
    omegam_fid: float
    rdrag_fid: float
    H0_fid: float

    @classmethod
    def from_cosmo(
        cls,
        omegam: float,
        rdrag: float,
        H0: float,
        sigma8: float,
        zeff: float,
        omegam_fid: float,
        rdrag_fid: float,
        H0_fid: float,
    ):
        rdrag_hubble_unit = rdrag * H0 / 100
        DH_over_rdrag_fid = DH(zeff, omegam_fid) / (rdrag_fid * H0_fid / 100)
        DM_over_rdrag_fid = DM(zeff, omegam_fid) / (rdrag_fid * H0_fid / 100)
        alpara = (DH(zeff, omegam) / rdrag_hubble_unit) / DH_over_rdrag_fid
        alperp = (DM(zeff, omegam) / rdrag_hubble_unit) / DM_over_rdrag_fid
        fsigma8 = growth_rate(zeff, omegam) * growth_factor(zeff, omegam) * sigma8
        return cls(alpara, alperp, fsigma8, zeff, omegam_fid, rdrag_fid, H0_fid)

    def h_fid(self):
        return self.H0_fid / 100

    def alpha(self) -> float:
        """dilation / isotropic BAO parameter"""
        return (self.alpara * self.alperp**2) ** (1 / 3)

    def epsilon(self) -> float:
        """warping / aniostropic BAO parameter"""
        return (self.alpara / self.alperp) ** (1 / 3) - 1

    def DH_over_rdrag(self) -> float:
        """D_H / r_{drag}"""
        return self.alpara * (DH(self.zeff, self.omegam_fid) / (self.rdrag_fid * self.h_fid()))

    def DM_over_rdrag(self) -> float:
        """D_M / r_{drag}"""
        return self.alperp * (DM(self.zeff, self.omegam_fid) / (self.rdrag_fid * self.h_fid()))

    def DV_over_rdrag(self) -> float:
        """D_V / r_{drag}"""
        return (self.zeff * self.DM_over_rdrag() ** 2 * self.DH_over_rdrag()) ** (1 / 3)

    def H0rdrag(self):
        omegam = self.omegam()
        return 299792.458 / efunc(self.zeff, omegam) / self.DH_over_rdrag()

    def H0(self, rdrag: float | None = None):
        """H0 in km/s/Mpc

        Parameters
        ----------
        rdrag: float, optional
            r_drag in Mpc unit, default to rdrag_fid
        """
        if rdrag is None:
            rdrag = self.rdrag_fid
        return self.H0rdrag() / rdrag

    @cached_property
    def _omegam(self) -> float:
        # fmt: off
        constant = (self.alpara / self.alperp) * DH(self.zeff, self.omegam_fid) / DM(self.zeff, self.omegam_fid)
        # fmt: on
        return fsolve(lambda om: DH(self.zeff, om) / DM(self.zeff, om) - constant, 0.31)[0]

    def omegam(self):
        return self._omegam

    def sigma8(self):
        omegam = self.omegam()
        gz = growth_factor(self.zeff, omegam)
        fz = growth_rate(self.zeff, omegam)
        return self.fsigma8 / (fz * gz)


# TODO: check the correctness of this implementation
@dataclass
class KaiserModel:
    """Kaiser model with PyBird-like counter terms"""

    k: ndarrayf
    Plin: ndarrayf
    ells: tuple[int, ...] = (0, 2, 4)
    km: float = 0.7
    kr: float = 0.25
    nd: float = 1e-3
    default_params: dict[str, float | None] = field(
        default_factory=lambda: {
            "b1": None,
            "f": None,
            "cct": 0.0,
            "cr1": 0.0,
            "cr2": 0.0,
            "Pshot": 0.0,
        }
    )

    def __post_init__(self):
        for ell in self.ells:
            if ell % 2 != 0:
                raise ValueError(f"ells must all be even, got {self.ells}")

    def params(self, params_values_dict: dict[str, float]):
        updated_params: dict[str, float] = {}
        for k, v in self.default_params.items():
            if (value := params_values_dict.get(k, v)) is None:
                raise ValueError(f"missing parameter: {k}")
            updated_params[k] = value
        return updated_params

    def Plk_linear_grid(self, params_values_dict: dict[str, float]) -> ndarrayf:
        params = self.params(params_values_dict)
        b1, f = params["b1"], params["f"]
        coef = [
            b1**2 + 2 / 3 * b1 * f + 1 / 5 * f**2,
            4 / 3 * b1 * f + 4 / 7 * f**2,
            8 / 35 * f**2,
        ]
        return np.outer(coef, self.Plin)

    def Plk_counter_grid(self, params_values_dict: dict[str, float]) -> ndarrayf:
        params = self.params(params_values_dict)
        b1, f, cct, cr1, cr2 = (
            params["b1"],
            params["f"],
            params["cct"],
            params["cr1"],
            params["cr2"],
        )
        cct /= self.km**2
        cr1 /= self.kr**2
        cr2 /= self.kr**2
        # fmt: off
        coef = [
            b1*cct + 1/3*b1*cr1 + 1/5*b1*cr2 + 1/3*cct*f + 1/5*cr1*f+ 1/7*cr2*f,
            2/3*b1*cr1 + 4/7*b1*cr2 + 2/3*cct*f + 4/7*cr1*f + 10/21*cr2*f,
            8/35*b1*cr2 + 8/35*cr1*f + 24/77*cr2*f,
        ]
        # fmt: on
        return np.outer(coef, self.k**2 * self.Plin)

    def Plk_stochastic_grid(self, params_values_dict: dict[str, float]) -> ndarrayf:
        retval = np.zeros((3, len(self.k)))
        retval[0] = self.params(params_values_dict)["Pshot"] / self.nd
        return retval

    def Plk_grid(self, params_values_dict: dict[str, float]) -> ndarrayf:
        # XXX: workaround
        return (
            self.Plk_linear_grid(params_values_dict)
            + self.Plk_counter_grid(params_values_dict)
            + self.Plk_stochastic_grid(params_values_dict)
        )[[(0, 2, 4).index(ell) for ell in self.ells]]

    def Plk(self, params_values_dict: dict[str, float], ells, k) -> ndarrayf:
        is_1d = False if isinstance(ells, Iterable) else True
        ells = np.atleast_1d(ells)  # type: ignore
        if not set(ells).issubset(set(self.ells)):
            raise ValueError(f"ells must be subset of {self.ells}")
        k = np.atleast_1d(k)
        grid = self.Plk_grid(params_values_dict) * self.k[None, :]
        grid = np.insert(grid, 0, 0, axis=1)
        retval = interp1d(np.hstack([0, self.k]), grid, axis=-1, kind="cubic")(k) / k[None, :]
        retval = np.array([retval[self.ells.index(ell)] for ell in ells])
        return retval[0] if is_1d else retval

    @property
    def Plinear_interpolator(self):
        x = np.hstack([0.0, self.k])
        y = np.hstack([0.0, self.k * self.Plin])
        tmp = interp1d(x, y, kind="cubic")
        fn = lambda k: tmp(k) / k
        return fn

    def fit(
        self,
        ells: int | Sequence[int],
        krange: tuple[float, float],
        multipole: Multipole,
        covariance: ndarrayf | None = None,
        varied_params: Sequence[str] = ["b1", "cct", "Pshot"],
        initial_point: ndarrayf | None = None,
    ) -> dict[str, float]:
        kmin, kmax = krange
        ells = tuple(ells) if isinstance(ells, Iterable) else (ells,)
        data = np.hstack([multipole[multipole.symbol + str(ell)][kmin:kmax] for ell in ells])
        ileft, iright = multipole.k.searchsorted([kmin, kmax])  # [kmin, kmax)
        k = multipole.k[ileft:iright]
        Plin = self.Plinear_interpolator(k)
        kaiser = type(self)(
            k=k,
            Plin=Plin,
            ells=ells,
            km=self.km,
            kr=self.kr,
            nd=self.nd,
            default_params=self.default_params.copy(),
        )

        if covariance is None:

            def fn(x):
                params = dict(zip(varied_params, x))
                return kaiser.Plk_grid(params).reshape(-1) - data

        else:
            invcov = np.linalg.inv(covariance)

            def fn(x):
                params = dict(zip(varied_params, x))
                res = kaiser.Plk_grid(params).reshape(-1) - data
                return np.array([res @ invcov @ res]) ** 0.5

        x0: Any = initial_point
        if initial_point is None:
            x0 = [kaiser.default_params[k] for k in varied_params]
            if any(v is None for v in x0):
                raise ValueError(
                    f"missing initial point for {varied_params}, " "please provide initial_point"
                )
            x0 = np.array(x0)
        return dict(zip(varied_params, least_squares(fn, x0).x))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ells={self.ells}, "
            f"kmin={self.k.min():.3f}, "
            f"kmax={self.k.max():.3f}, "
            f"km={self.km:.3f}, "
            f"kr={self.kr:.3f}, "
            f"nd={self.nd:.3e})"
        )
