from __future__ import annotations
import itertools
import logging
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, Mapping, TypeVar, TYPE_CHECKING, Union
from typing_extensions import TypeAlias
from cobaya import get_model
from cobaya.yaml import yaml_dump
from cobaya.yaml import yaml_dump_file
from cobaya.yaml import yaml_load_file
from .reader import read_pkl
from .tools import pairwise, verbose_guard

if TYPE_CHECKING:
    from .theory import PlkInterpolator
    from .typing import ndarrayf

FilePath: TypeAlias = Union[str, os.PathLike]
_T = TypeVar("_T")


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
            params_values_dict = {
                k: params_values_dict[k] for k in self.sampled_params()
            }
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

    def marginalized_params(
        self, with_marg_prefix: bool = False
    ) -> dict[str, dict[str, Any]]:
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


@dataclass
class Multipole(Mapping[str, pd.Series]):
    data: pd.DataFrame = field(repr=False, compare=False)
    k: ndarrayf = field(repr=False)
    ells: tuple[int, ...]
    symbol: str

    @classmethod
    def loadtxt(
        cls,
        path: FilePath,
        cov_path: FilePath | None = None,
        logger: logging.Logger | None = None,
    ):
        data = read_pkl(path, logger=logger)
        k = data["k"].to_numpy()
        pattern = re.compile(r"(?P<symbol>[A-Za-z]+)(?P<ell>\d+)")
        db = defaultdict(list)
        for name in data.columns:
            if match := pattern.match(name):
                db[match.group("symbol")].append(int(match.group("ell")))
        cnt = Counter({k: len(v) for k, v in db.items()})
        symbol = cnt.most_common(1)[0][0]
        ells = tuple(sorted(db[symbol]))
        multipole = cls(data, k, ells, symbol)
        if cov_path:
            multipole.apply_covariance(np.loadtxt(cov_path))
        return multipole

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

    def apply_covariance(self, cov: ndarrayf):
        sections = list(
            itertools.accumulate(
                self.values(), lambda total, elem: total + elem.size, initial=0
            )
        )[1:-1]
        errs = np.split(np.sqrt(cov.diagonal()), sections)
        for ell, yerr in zip(self.ells, errs):
            self.data[f"{self.symbol}{ell}err"] = yerr

    def plot(
        self, ax=None, errorbar_style: dict[str, Any] = {"fmt": ".", "capsize": 2}
    ):
        if ax is None:
            ax = plt.gca()
        k = self.k
        if (Pk := self.get(self.symbol + "4")) is not None:
            Pkerr = self.hex_err()
            Pkerr = None if Pkerr is None else k * Pkerr
            ax.errorbar(k, k * Pk, yerr=Pkerr, c="g", **errorbar_style)
        if (Pk := self.get(self.symbol + "2")) is not None:
            Pkerr = self.quad_err()
            Pkerr = None if Pkerr is None else k * Pkerr
            ax.errorbar(k, k * Pk, yerr=Pkerr, c="b", **errorbar_style)
        if (Pk := self.get(self.symbol + "0")) is not None:
            Pkerr = self.mono_err()
            Pkerr = None if Pkerr is None else k * Pkerr
            ax.errorbar(k, k * Pk, yerr=Pkerr, c="k", **errorbar_style)
        ax.set_xlabel(R"$k$ $[h\,\mathrm{Mpc}^{-1}]$")
        ax.set_ylabel(Rf"$k{self.symbol}_\ell(k)$ $[h^{-1}\,\mathrm{{Mpc}}]^2$")
        return ax

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
    from .likelihood import EFTLike

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
            multipole = Multipole.loadtxt(likelihood.data[tracer]["path"])
            tracer_multipoles[tracer] = multipole
            nsize.append(multipole.data_vector().size)
        for (istart, iend), tracer in zip(
            pairwise(itertools.accumulate(nsize, initial=0)), tracers
        ):
            subcov = full_covmat[istart:iend, istart:iend]
            tracer_multipoles[tracer].apply_covariance(subcov)
    return tracer_multipoles


@dataclass
class BestfitModel:
    yaml_file: str
    verbose: bool = False

    def __post_init__(self):
        from .likelihood import EFTLike

        # step 1: get global bestfit
        products = EFTLikeProducts.from_yaml_file(self.yaml_file)
        marg_param_prefix = products.marg_param_prefix()
        itrim = len(marg_param_prefix)
        bestfit = products.global_bestfit_if_possible()
        bestfit = {
            (k[itrim:] if k.startswith(marg_param_prefix) else k): v
            for k, v in bestfit.items()
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
            fullchi2.append(likename + "_fullchi2")
            ndata.append(likelihood.ndata)
            hartlap.append(likelihood.hartlap if likelihood.hartlap is not None else 1)
            requires[likename + "_fullchi2"] = None
        # step 3: evaluate full model
        fullinfo = products.dump_full_model_info()
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

    def Plk_interpolator(self, tracer: str) -> PlkInterpolator:
        return self.model.provider.get_nonlinear_Plk_interpolator(
            tracer, chained=self.chained[tracer]
        )

    def plot(
        self, tracer: str, ax=None, errorbar_style: dict[str, Any] = {"capsize": 2}
    ):
        if ax is None:
            ax = plt.gca()
        self.multipoles[tracer].plot(ax, errorbar_style)
        k = np.linspace(0.0005, 0.3, 1000)
        Plk = self.Plk_interpolator(tracer)
        if 0 in Plk.ls:
            ax.plot(k, k * Plk(0, k), c="k")
        if 2 in Plk.ls:
            ax.plot(k, k * Plk(2, k), c="b")
        if 4 in Plk.ls:
            ax.plot(k, k * Plk(4, k), c="g")
        ax.set_title(tracer.replace("_", " "))
        return ax
