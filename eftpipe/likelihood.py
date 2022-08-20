from __future__ import annotations
import numpy as np
from typing import Any
from cobaya.likelihood import Likelihood
from .lssdata import LSSData
from .marginal import Marginalizable


def flatten(ls, ls_tot, array):
    idx = [ls_tot.index(ell) for ell in ls]
    return array[idx, :].reshape(-1)


class EFTLikeSingle(Likelihood, Marginalizable):
    """EFT likelihood for single tracer
    """

    file_base_name = "eftlike_s"

    tracer: str
    prefix: str
    lssdata: LSSData
    chained: bool
    with_binning: bool
    binning: dict[str, Any]
    marg: dict[str, dict[str, Any]]

    def initialize(self) -> None:
        super().initialize()
        self.lssdata = LSSData.from_dict(self.lssdata)  # type: ignore
        # assume all masked ks are the same
        self.kout = self.lssdata.fullshape[0][0].x
        self.binning = self.binning or {}
        self.binning = {"kout": self.kout, **self.binning}
        self.ls = self.lssdata.fullshape[0].ls
        if self.marg:
            self.setup_prior(self.marg)

    def get_requirements(self):
        reqs = {}
        if self.with_binning:
            reqs["nonlinear_Plk_grid"] = {
                self.tracer: {
                    "ls": self.ls,
                    "chained": self.chained,
                    "binned": self.with_binning,
                    "binning": self.binning,
                }
            }
        else:
            reqs["nonlinear_Plk_interpolator"] = {
                self.tracer: {"ls": self.ls, "chained": self.chained,}
            }
        if self.marg:
            reqs["nonlinear_Plk_gaussian_grid"] = {
                self.tracer: {
                    "ls": self.ls,
                    "chained": self.chained,
                    "binned": self.with_binning,
                    **({"binning": self.binning} if self.with_binning else {}),
                }
            }
        return reqs

    # override
    def marginalizable_params(self) -> list[str]:
        return [
            self.prefix + name
            for name in ("b3", "cct", "cr1", "cr2", "ce0", "cemono", "cequad")
        ]

    # override
    def PG(self):
        ls_tot, _, table = self.provider.get_nonlinear_Plk_gaussian_grid(
            self.tracer, chained=self.chained, binned=self.with_binning
        )
        out = []
        for bG in self.valid_prior.keys():
            out.append(flatten(self.ls, ls_tot, table[bG]))
        return np.vstack(out)

    # override
    def PNG(self):
        if self.with_binning:
            ls_tot, _, plk = self.provider.get_nonlinear_Plk_grid(
                self.tracer, chained=self.chained, binned=self.with_binning
            )
            out = flatten(self.ls, ls_tot, plk)
        else:
            fn = self.provider.get_nonlinear_Plk_interpolator(
                self.tracer, chained=self.chained
            )
            out = fn(self.ls, self.kout).reshape(-1)
        return out

    def calculate(self, state, want_derived=True, **params_values_dict):
        if self.marg:
            state["logp"] = self.marginalized_logp(
                self.lssdata.data_vector, self.lssdata.invcov
            )
        else:
            res = self.lssdata.data_vector - self.PNG()
            chi2 = res @ self.lssdata.invcov @ res
            state["logp"] = -0.5 * chi2

