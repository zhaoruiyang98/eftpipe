from __future__ import annotations
import inspect
import time
import numpy as np
from typing import Any, TYPE_CHECKING
from cobaya.log import HasLogger
from cobaya.log import LoggedError

if TYPE_CHECKING:
    from .etyping import ndarrayf


def eval_callable(s: str, globals: dict[str, Any]):
    fn = eval(s, globals)
    # support positional arguments only
    # NOTE: loc and scale of gaussian parameters should only depend on nongaussian parameters,
    # though this is not enforced
    argnames = inspect.getfullargspec(fn).args
    return fn(*(globals[p] for p in argnames))


def valid_prior_config(config: Any) -> bool:
    if config is None:
        return True
    if isinstance(config, dict):
        if "loc" in config or "scale" in config:
            return True
    return False


class Marginalizable(HasLogger if TYPE_CHECKING else object):
    """Marginalizable mixin

    Notes
    -----
    * Marginalizable should be inherited (indirectly) from HasLogger
    * valid_prior will be sorted according to the order of parameters appearing in marginalizable_params
    """

    valid_prior: dict[str, dict[str, float | str]]

    def marginalizable_params(self) -> list[str]:
        raise NotImplementedError

    def PG(self) -> ndarrayf:
        raise NotImplementedError

    def PNG(self) -> ndarrayf:
        raise NotImplementedError

    def get_data_vector(self) -> ndarrayf:
        raise NotImplementedError

    def get_invcov(self) -> ndarrayf:
        raise NotImplementedError

    def env(self) -> dict[str, Any]:
        return {"np": np}

    @property
    def mu_G(self) -> ndarrayf:
        env = self.env()
        loc_it = [dct["loc"] for dct in self.valid_prior.values()]
        return np.array(
            [eval_callable(x, env) if isinstance(x, str) else x for x in loc_it],
            dtype=np.float64,
        )

    @property
    def sigma_inv(self) -> ndarrayf:
        env = self.env()
        std_it = (dct["scale"] for dct in self.valid_prior.values())
        std = [eval_callable(x, env) if isinstance(x, str) else x for x in std_it]
        if np.inf in std:
            return np.zeros((len(std), len(std)))
        np.fill_diagonal(self._sigma_inv, 1 / np.array(std, dtype=np.float64) ** 2)
        return self._sigma_inv

    def marginalized_logp(
        self, return_bGbest: bool = False, jeffreys: bool = False
    ) -> float | tuple[float, float, dict[str, float]]:
        R"""calculate marginalized posterior

        Parameters
        ----------
        return_bGbest : bool
            whether to return fullchi2 and bestfit bG parameters dict, default False
        jeffreys : bool
            whether to use Jeffreys prior, default False

        Returns
        -------
        float
            marginalized log-posterior
        float | tuple[float, float, dict[str, float]]
            marginalized log-posterior, fullchi2 and bestfit bG parameters dict

        Notes
        -----
        .. math::
            -2\ln\mathcal{P}_\mathrm{mag} = -F_{1,i} (F_2^{-1})_{ij} F_{1,j} + F_0 + \ln\det\left(\frac{F_2}{2\pi}\right)
        """
        start = time.perf_counter()
        dvector, invcov = self.get_data_vector(), self.get_invcov()
        PNG = self.PNG()
        PG = self.PG()
        mu_G, sigma_inv = self.mu_G, self.sigma_inv
        # XXX: possible cache?
        F2ij = self.calc_F2ij(PG, invcov, sigma_inv)
        F1i = self.calc_F1i(PG, PNG, invcov, dvector, mu_G, sigma_inv)
        F0 = self.calc_F0(PNG, invcov, dvector, mu_G, sigma_inv)
        sign, logdet = np.linalg.slogdet(F2ij / (2 * np.pi))
        if sign <= 0:
            # TODO: svd analysis
            self.mpi_warning("det of F2ij <= 0")
            raise RuntimeError("det of F2ij <= 0")
        bGbest = np.linalg.solve(F2ij, F1i)
        chi2: float
        if jeffreys:
            chi2 = -F1i @ bGbest + F0  # type: ignore
        else:
            chi2 = -F1i @ bGbest + F0 + logdet

        if not return_bGbest:
            end = time.perf_counter()
            self.mpi_debug("marginalized_logp: time used: %s", end - start)
            return -0.5 * chi2

        # TODO: bGbest w/o sigma_inv
        Pall = bGbest @ PG + PNG
        res = Pall - dvector
        fullchi2 = np.einsum("i,ij,j->", res, invcov, res, optimize=True)
        retval = (
            -0.5 * chi2,
            fullchi2,
            {bG: val for bG, val in zip(self.valid_prior.keys(), bGbest)},
        )
        end = time.perf_counter()
        self.mpi_debug("marginalized_logp: time used: %s", end - start)
        return retval

    def setup_prior(self, prior: dict[str, dict[str, Any]]) -> None:
        """setup self.valid_prior, self.mu_G and self.sigma_inv"""
        self.valid_prior = self.update_prior(prior)
        nmarg = len(self.valid_prior)
        self._sigma_inv = np.zeros((nmarg, nmarg))

    def report_marginalized(self) -> None:
        self.mpi_info("the following parameters are marginalized with gaussian prior:")
        for name, dct in self.valid_prior.items():
            self.mpi_info(f"{name}:")
            self.mpi_info(f"  loc: {dct['loc']}")
            self.mpi_info(f"  scale: {dct['scale']}")

    def bG_bestfit(self, F1i=None, F2ij=None) -> dict[str, float]:
        """helper method to extract bestfit bG parameters"""
        if F1i is None or F2ij is None:
            PNG = self.PNG()
            PG = self.PG()
            mu_G, sigma_inv = self.mu_G, self.sigma_inv
            dvector, invcov = self.get_data_vector(), self.get_invcov()
            F1i = self.calc_F1i(PG, PNG, invcov, dvector, mu_G, sigma_inv)
            F2ij = self.calc_F2ij(PG, invcov, sigma_inv)
        ret = np.linalg.solve(F2ij, F1i)
        return {bG: val for bG, val in zip(self.valid_prior.keys(), ret)}

    def calc_F2ij(self, PG, invcov, sigma_inv) -> ndarrayf:
        R"""calculate F2 matrix

        Notes
        -----
        .. math::
            F_{2,ij} = P_{G,\alpha}^i C_{\alpha\beta}^{-1} P_{G,\beta}^j + \sigma_{ij}^-1
        """
        return np.einsum("ia,ab,jb->ij", PG, invcov, PG, optimize=True) + sigma_inv

    def calc_F1i(self, PG, PNG, invcov, dvector, mu_G, sigma_inv) -> ndarrayf:
        R"""calculate F1 vector

        Notes
        -----
        .. math::
            F_{1,i} = -P_{G,\alpha}^i C_{\alpha\beta}^{-1} (P_{NG,\aplha} - D_{\alpha}) + \sigma_{ij}^{-1} \mu_{G,j}
        """
        return (
            -np.einsum("ia,ab,b->i", PG, invcov, PNG - dvector, optimize=True)
            + sigma_inv @ mu_G
        )

    def calc_F0(self, PNG, invcov, dvector, mu_G, sigma_inv) -> ndarrayf:
        R"""calculate F0

        Notes
        -----
        .. math::
            F_0 = (P_{NG,\alpha} - D_{\alpha}) C_{\alpha\beta}^{-1} (P_{NG,\beta} - D_\beta) + \mu_{G,i}\sigma_{ij}^{-1}\mu_{G,j}
        """
        res = PNG - dvector
        return res @ invcov @ res + mu_G @ sigma_inv @ mu_G

    def update_prior(
        self, prior: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, float | str]]:
        """update prior to standard form and sort it"""
        marginalizable_params = self.marginalizable_params()
        for key in prior.keys():
            if key not in marginalizable_params:
                raise LoggedError(
                    self.log,
                    "key <%s> is not marginalizable",
                    key,
                )
        newdct = {}
        nscale_inf = 0
        for name, dct in prior.items():
            if dct is None:
                loc, scale = None, None
            else:
                loc = dct.get("loc", None)
                scale = dct.get("scale", None)
            if scale is None or scale == np.inf:
                scale = np.inf
                nscale_inf += 1
            if loc is None:
                loc = 0
            newdct[name] = {"loc": loc, "scale": scale}
        idx = [marginalizable_params.index(name) for name in newdct.keys()]
        outdct = {
            name: dct
            for _, dct, name in sorted(
                zip(idx, newdct.values(), newdct.keys()), key=lambda t: t[0]
            )
        }
        if nscale_inf != 0 and nscale_inf != len(outdct):
            raise LoggedError(
                self.log,
                "only support setting infinite scale for all parameters",
            )
        return outdct
