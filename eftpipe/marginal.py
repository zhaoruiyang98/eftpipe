import numpy as np
from numpy import ndarray as NDArray
from typing import (
    Dict,
    Tuple,
    Any,
)
from eftpipe.typing import GaussianData, VectorTheory, LogFunc


class MargGaussian:
    def __init__(
        self,
        data_obj: GaussianData,
        theory_obj: VectorTheory,
        prior: Dict[str, Any],
        logfunc: LogFunc = print
    ) -> None:
        self.data_obj = data_obj
        self.theory_obj = theory_obj
        self.invcov = data_obj.invcov
        self.D = data_obj.data_vector
        valid_prior = self._update_prior(prior)
        logfunc('==========================>')
        logfunc(f'the following parameters are marginalized with gaussian prior:')
        for name, dct in valid_prior.items():
            logfunc(f'{name}:')
            logfunc(f"  loc: {dct['loc']}")
            logfunc(f"  scale: {dct['scale']}")
        logfunc('<==========================')
        self.theory_obj.set_marg(valid_prior)
        self.mu_G, self.sigma_inv = self._calc_prior(valid_prior)

    def _update_prior(self, prior: Dict[str, Any]) -> Dict[str, Any]:
        can_marg_params = self.theory_obj.can_marg_params()
        for key in prior.keys():
            if key not in can_marg_params:
                raise ValueError(
                    f"key <{key}> not found in VectorTheory's can_marg_params")
        newdct = {}
        nscale_inf = 0
        for name, dct in prior.items():
            if dct is None:
                loc, scale = None, None
            else:
                loc = dct.get('loc', None)
                scale = dct.get('scale', None)
            if scale is None or scale == np.inf:
                scale = np.inf
                nscale_inf += 1
            if loc is None:
                loc = 0
            newdct[name] = {'loc': loc, 'scale': scale}
        inds = [can_marg_params.index(name) for name in newdct.keys()]
        outdct = {
            name: dct
            for _, dct, name in sorted(
                zip(inds, newdct.values(), newdct.keys()), key=lambda t: t[0])
        }
        if nscale_inf != 0 and nscale_inf != len(can_marg_params):
            raise ValueError(
                f"only support set infinite scale for all parameters")
        return outdct

    def _calc_prior(self, prior: Dict[str, Any]) -> Tuple[NDArray, NDArray]:
        mu_G = [dct['loc'] for dct in prior.values()]
        mu_G = np.array(mu_G)
        nG = len(mu_G)
        std = [dct['scale'] for dct in prior.values()]
        if np.inf in std:
            sigma_inv = np.zeros((nG, nG))
        else:
            sigma_inv = np.diag(1 / np.array(std)**2)
        return mu_G, sigma_inv

    def calc_F2ij(self, PG) -> NDArray:
        r"""calculate F2 matrix
        
        Notes
        -----
        .. math::
            F_{2,ij} = P_{G,\alpha}^i C_{\alpha\beta}^{-1} P_{G,\beta}^j + \sigma_{ij}^-1
        """
        return PG @ self.invcov @ PG.T + self.sigma_inv

    def calc_F1i(self, PG, PNG) -> NDArray:
        r"""calculate F1 vector
        
        Notes
        -----
        .. math::
            F_{1,i} = -P_{G,\alpha}^i C_{\alpha\beta}^{-1} (P_{NG,\aplha} - D_{\alpha}) + \sigma_{ij}^{-1} \mu_{G,j}
        """
        return -PG @ self.invcov @ (PNG - self.D) + self.sigma_inv @ self.mu_G

    def calc_F0(self, PNG) -> NDArray:
        r"""calculate F0
        
        Notes
        -----
        .. math::
            F_0 = (P_{NG,\alpha} - D_{\alpha}) C_{\alpha\beta}^{-1} (P_{NG,\beta} - D_\beta) + \mu_{G,i}\sigma_{ij}^{-1}\mu_{G,j}
        """
        res = PNG - self.D
        return res @ self.invcov @ res + self.mu_G @ self.sigma_inv @ self.mu_G

    def calculate(self, all_params_dict: Dict[str, Any]) -> float:
        r"""calculate marginalized posterior
        
        Notes
        -----
        .. math::
            -2\ln\mathcal{P}_\mathrm{mag} = -F_{1,i} (F_2^{-1})_{ij} F_{1,j} + F_0 + \ln\det\left(\frac{F_2}{2\pi}\right)
        """
        PNG = self.theory_obj.PNG(all_params_dict)
        PG = self.theory_obj.PG(all_params_dict)
        F2ij = self.calc_F2ij(PG)
        F1i = self.calc_F1i(PG, PNG)
        F0 = self.calc_F0(PNG)
        det = np.linalg.det(F2ij / (2 * np.pi))
        if det < 0:
            raise ValueError(f"det of F2ij < 0")
        chi2 = (
            - F1i @ np.linalg.inv(F2ij) @ F1i
            + F0
            + np.log(det)
        )
        return -0.5 * chi2
