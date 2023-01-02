from __future__ import annotations
import time
import numpy as np
from typing import Any, TYPE_CHECKING
from cobaya.log import HasLogger
from cobaya.log import LoggedError

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Marginalizable(HasLogger if TYPE_CHECKING else object):
    """Marginalizable mixin

    Notes
    -----
    Marginalizable should be inherited (indirectly) from HasLogger
    """

    valid_prior: dict[str, dict[str, float]]
    mu_G: NDArray
    sigma_inv: NDArray

    def marginalizable_params(self) -> list[str]:
        raise NotImplementedError

    def PG(self) -> NDArray:
        raise NotImplementedError

    def PNG(self) -> NDArray:
        raise NotImplementedError

    def get_data_vector(self) -> NDArray:
        raise NotImplementedError

    def get_invcov(self) -> NDArray:
        raise NotImplementedError

    def marginalized_logp(self) -> float:
        R"""calculate marginalized posterior

        Parameters
        ----------
        dvector : NDArray, 1d
            data vector
        invcov : NDArray, 2d
            inverse covariance matrix of data vector

        Returns
        -------
        float
            marginalized log-posterior

        Notes
        -----
        .. math::
            -2\ln\mathcal{P}_\mathrm{mag} = -F_{1,i} (F_2^{-1})_{ij} F_{1,j} + F_0 + \ln\det\left(\frac{F_2}{2\pi}\right)
        """
        start = time.perf_counter()
        dvector, invcov = self.get_data_vector(), self.get_invcov()
        PNG = self.PNG()
        PG = self.PG()
        # XXX: possible cache?
        F2ij = self.calc_F2ij(PG, invcov)
        F1i = self.calc_F1i(PG, PNG, invcov, dvector)
        F0 = self.calc_F0(PNG, invcov, dvector)
        sign, logdet = np.linalg.slogdet(F2ij / (2 * np.pi))
        if sign <= 0:
            raise RuntimeError(
                "det of F2ij <= 0, please consider tighter prior on gaussian parameters"
            )
        chi2 = -F1i @ np.linalg.solve(F2ij, F1i) + F0 + logdet

        end = time.perf_counter()
        self.mpi_debug("marginalized_logp: time used: %s", end - start)
        return -0.5 * chi2

    def setup_prior(self, prior: dict[str, dict[str, Any]]) -> None:
        """setup self.valid_prior, self.mu_G and self.sigma_inv"""
        self.valid_prior = self._update_prior(prior)
        self.mu_G, self.sigma_inv = self._calc_prior(self.valid_prior)

    def report_marginalized(self) -> None:
        self.mpi_info("the following parameters are marginalized with gaussian prior:")
        for name, dct in self.valid_prior.items():
            self.mpi_info(f"{name}:")
            self.mpi_info(f"  loc: {dct['loc']}")
            self.mpi_info(f"  scale: {dct['scale']}")

    def bG_bestfit(self) -> dict[str, float]:
        """helper method to extract bestfit bG parameters"""
        PNG = self.PNG()
        PG = self.PG()
        dvector, invcov = self.get_data_vector(), self.get_invcov()
        F1i = self.calc_F1i(PG, PNG, invcov, dvector)
        F2ij = self.calc_F2ij(PG, invcov)
        ret = np.linalg.inv(F2ij) @ F1i
        return {bG: val for bG, val in zip(self.valid_prior.keys(), ret)}

    def calc_F2ij(self, PG, invcov) -> NDArray:
        R"""calculate F2 matrix

        Notes
        -----
        .. math::
            F_{2,ij} = P_{G,\alpha}^i C_{\alpha\beta}^{-1} P_{G,\beta}^j + \sigma_{ij}^-1
        """
        return PG @ invcov @ PG.T + self.sigma_inv

    def calc_F1i(self, PG, PNG, invcov, dvector) -> NDArray:
        R"""calculate F1 vector

        Notes
        -----
        .. math::
            F_{1,i} = -P_{G,\alpha}^i C_{\alpha\beta}^{-1} (P_{NG,\aplha} - D_{\alpha}) + \sigma_{ij}^{-1} \mu_{G,j}
        """
        return -PG @ invcov @ (PNG - dvector) + self.sigma_inv @ self.mu_G

    def calc_F0(self, PNG, invcov, dvector) -> NDArray:
        R"""calculate F0

        Notes
        -----
        .. math::
            F_0 = (P_{NG,\alpha} - D_{\alpha}) C_{\alpha\beta}^{-1} (P_{NG,\beta} - D_\beta) + \mu_{G,i}\sigma_{ij}^{-1}\mu_{G,j}
        """
        res = PNG - dvector
        return res @ invcov @ res + self.mu_G @ self.sigma_inv @ self.mu_G

    def _update_prior(
        self, prior: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
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
        if nscale_inf != 0 and nscale_inf != len(marginalizable_params):
            raise LoggedError(
                self.log,
                "only support setting infinite scale for all parameters",
            )
        return outdct

    def _calc_prior(self, prior: dict[str, Any]) -> tuple[NDArray, NDArray]:
        mu_G = [dct["loc"] for dct in prior.values()]
        mu_G = np.array(mu_G)
        nG = len(mu_G)
        std = [dct["scale"] for dct in prior.values()]
        if np.inf in std:
            sigma_inv = np.zeros((nG, nG))
        else:
            sigma_inv = np.diag(1 / np.array(std) ** 2)
        return mu_G, sigma_inv
