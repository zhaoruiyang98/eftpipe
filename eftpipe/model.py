from __future__ import annotations
import math
import tempfile
import weakref
from copy import deepcopy
from typing import Any, Literal, TYPE_CHECKING
from cobaya.model import get_model
from .tools import disable_logging
from .tools import do_nothing

if TYPE_CHECKING:
    from .theory import PlkInterpolator


class EFTModel:
    """A simplified interface to compute EFTofLSS power spectrum

    Parameters
    ----------
    z : float
        redshift
    ndA : float
        number density of tracer A
    ndB : float | None
        number density of tracer B, if None, compute the auto power spectrum of tracer A
    kmA : float
        nonlinear scale of tracer A
    krA : float
        renormalization scale of tracer A
    kmB : float
        nonlinear scale of tracer B
    krB : float
        renormalization scale of tracer B
    cache_dir_path : str, optional
        path to cache directory, default to temporary directory
    use_cb : bool
        if true, use the linear power spectrum of cdm + baryon as input, default True
    IRcutoff : "all", "loop", "resum" or bool
    kIR : float, optional

    Examples
    --------
    >>> from eftpipe.model import EFTModel
    >>> model = EFTModel(0.845, 1e-4).set_cosmology().set_IRresum().done(ellmax=4)
    >>> plk = model(1.59, 1.26, 1.67, 0, -7.1, -2.6, -0.36, 0.016, 0, -0.041)
    """

    def __init__(
        self,
        z: float,
        ndA: float = 1e-4,
        ndB: float | None = None,
        kmA: float = 0.7,
        krA: float = 0.25,
        kmB: float = 0.7,
        krB: float = 0.25,
        cache_dir_path: str | None = None,
        use_cb: bool = True,
        with_RSD: bool = True,
        IRcutoff: Literal["all", "loop", "resum"] | bool = False,
        kIR: float | None = None,
    ):
        self._done = False
        self.z = z
        self.use_cb = use_cb
        self.theory: dict[str, Any] = {"eftpipe.classynu": None, "eftpipe.eftlss": {}}
        self.params: dict[str, Any] = {}
        self.likelihood = {"one": None}
        if not cache_dir_path:
            tempdir = tempfile.TemporaryDirectory()
            cache_dir_path = tempdir.name
            weakref.finalize(self, tempdir.cleanup)
        self.theory["eftpipe.eftlss"]["cache_dir_path"] = cache_dir_path
        self.cross = ndB is not None
        if self.cross:
            self.theory["eftpipe.eftlss"]["tracers"] = {
                "A": {"prefix": "A_", "z": z, "nd": ndA, "km": kmA, "kr": krA},
                "B": {"prefix": "B_", "z": z, "nd": ndB, "km": kmB, "kr": krB},
                "x": {"prefix": "x_", "z": z, "cross": ["A", "B"]},
            }
        else:
            self.theory["eftpipe.eftlss"]["tracers"] = {
                "x": {"prefix": "x_", "z": z, "nd": ndA, "km": kmA, "kr": krA}
            }
        self.theory["eftpipe.eftlss"]["tracers"]["default"] = {
            "provider": "classynu",
            "use_cb": use_cb,
            "with_IRresum": False,
            "IRcutoff": IRcutoff,
            "kIR": kIR,
            "with_RSD": with_RSD,
        }
        d = self.params
        if self.cross:
            common = {"prior": {"min": -10000, "max": 10000}, "ref": 0.0}
            d["A_b2"] = {"value": "lambda A_c2, A_c4: (A_c2 + A_c4) / np.sqrt(2.)"}
            d["A_b4"] = {"value": "lambda A_c2, A_c4: (A_c2 - A_c4) / np.sqrt(2.)"}
            d["B_b2"] = {"value": "lambda B_c2, B_c4: (B_c2 + B_c4) / np.sqrt(2.)"}
            d["B_b4"] = {"value": "lambda B_c2, B_c4: (B_c2 - B_c4) / np.sqrt(2.)"}
            # fmt: off
            for name in [
                "A_b1", "A_c2", "A_b3", "A_c4", "A_cct", "A_cr1", "A_cr2", 
                "B_b1", "B_c2", "B_b3", "B_c4", "B_cct", "B_cr1", "B_cr2",
                "x_ce0", "x_cemono", "x_cequad",
            ]:
                d[name] = deepcopy(common)
                if name in ["A_c2", "A_c4", "B_c2", "B_c4"]:
                    d[name]["drop"] = True
            # fmt: on
        else:
            common = {"prior": {"min": -10000, "max": 10000}, "ref": 0.0}
            d["x_b2"] = {"value": "lambda x_c2, x_c4: (x_c2 + x_c4) / np.sqrt(2.)"}
            d["x_b4"] = {"value": "lambda x_c2, x_c4: (x_c2 - x_c4) / np.sqrt(2.)"}
            for name in [
                "x_b1",
                "x_c2",
                "x_b3",
                "x_c4",
                "x_cct",
                "x_cr1",
                "x_cr2",
                "x_ce0",
                "x_cemono",
                "x_cequad",
            ]:
                d[name] = deepcopy(common)
                if name in ["x_c2", "x_c4"]:
                    d[name]["drop"] = True

    def set_cosmology(
        self,
        logA: float = 3.05,
        ns: float = 0.965,
        H0: float = 67.77,
        omegabh2: float = 0.0224,
        omegach2: float = 0.12,
        mnu: float | None = 0.06,
        neutrino_hierarchy: Literal["degenerate", "normal", "inverted"] = "degenerate",
        tau_reio: float = 0.055,
    ):
        self.params["logA"] = {"value": logA, "drop": True}
        self.params["As"] = {"value": "lambda logA: 1e-10*np.exp(logA)"}
        self.params["ns"] = ns
        self.params["H0"] = H0
        self.params["omegabh2"] = omegabh2
        self.params["omegach2"] = omegach2
        if mnu is None:
            self.theory["eftpipe.classynu"] = {
                "extra_args": {
                    "neutrino_hierarchy": "degenerate",
                    "N_ncdm": 0,
                    "N_ur": 3.046,
                }
            }
        else:
            self.params["mnu"] = mnu
            self.theory["eftpipe.classynu"] = {
                "extra_args": {"neutrino_hierarchy": neutrino_hierarchy}
            }
        self.params["tau_reio"] = tau_reio
        return self

    def set_IRresum(self, NFFT: int = 192):
        """
        Parameters
        ----------
        NFFT: int
            FFTLog nodes, deafult 192
        """
        d = self.theory["eftpipe.eftlss"]["tracers"]["x"]
        d["with_IRresum"] = True
        d["IRresum"] = {"NFFT": NFFT}
        return self

    def set_window(
        self,
        window_fourier_file: str | None = None,
        window_configspace_file: str | None = None,
        Na: int | None = None,
        Nl: int | None = None,
        Nq: int | None = 3,
        pmax: float = 0.3,
        accboost: int = 1,
        withmask: bool = True,
        windowk: float = 0.05,
        Nmax: int = 4096,
        xmin_factor: float = 1.0,
        xmax_factor: float = 100.0,
        bias: float = -1.6,
        window_param: float | None = 1,
        window_st: bool = True,
    ):
        """
        Parameters
        ----------
        window_fourier_file : str, optional
            path to fourier transformed window matrix,
            default None: always recompute the window and will not save to disk
        window_configspace_file : str, optional
            path to configspace window file, the header should be s, Q0, Q2, Q4, ...
            default None: load fourier matrix from disk
        Na : int, optional
            output multipoles, default co.Nl
        Nl : int, optional
            input multipoles, default co.Nl
        Nq : int, optional
            use Nq Qs, default 3, i.e. use Q0, Q2, Q4
        pmax : float
            integrate window up to pmax, default 0.3
        accboost : int
            accuracy boost x times of p grid, default 1
        withmask : bool
            set it True to mask window, default True
        windowk : float
            mask width, default 0.05, i.e. when k=0.1, only p in [0.1-0.05, 0.1+0.05] will be integrated
        Nmax : int
            FFTLog nodes, deafult 4096
        xmin_factor: float
            FFTLog xmin = smin * xmin_factor, where smin is determined by window_configspace_file
            default xmin_factor = 1.0
        xmax_factor: float
            FFTLog xmax = smax * xmax_factor, where smax is determined by window_configspace_file
            default xmax_factor = 100.
        bias: float
            FFTLog bias, default -1.6
        window_param: float | None
            window parameter, default 1
        window_st: bool
            set it True to apply window effect to stochastic terms
        """
        d = self.theory["eftpipe.eftlss"]["tracers"]["x"]
        d["with_window"] = True
        d["window"] = {
            "window_fourier_file": window_fourier_file,
            "window_configspace_file": window_configspace_file,
            "Na": Na,
            "Nl": Nl,
            "Nq": Nq,
            "pmax": pmax,
            "accboost": accboost,
            "withmask": withmask,
            "windowk": windowk,
            "Nmax": Nmax,
            "xmin_factor": xmin_factor,
            "xmax_factor": xmax_factor,
            "bias": bias,
            "window_param": window_param,
            "window_st": window_st,
        }
        return self

    def set_APeffect(
        self,
        Om_AP: float,
        z_AP: float | None = None,
        rdrag_AP: float | None = None,
        h_AP: float | None = None,
        nbinsmu: int = 200,
        accboost: int = 1,
        Nlmax: int | None = None,
        APst: bool = False,
    ):
        """
        Parameters
        ----------
        Om_AP : float
            omega matter for fiducial cosmology
        z_AP : float, optional
            fiducial effective redshift
        rdrag_AP : float, optional
            fiducial rdrag, in absolute units
        h_AP : float, optional
            fiducial h, used together with rdrag_AP,
            rdrag_AP and h_AP are only used in ``get_alperp_alpara``
        nbinsmu : int
            mu bins in [0, 1], default 200
        accboost : int
            boost the accuracy, default 1
        Nlmax : int, optional
            approximate Pkmu using Nlmax multipoles, default self.co.Nl
        APst : bool
            set it True to apply AP effect to stochastic terms
        """
        d = self.theory["eftpipe.eftlss"]["tracers"]["x"]
        d["with_APeffect"] = True
        d["APeffect"] = {
            "Om_AP": Om_AP,
            "z_AP": z_AP or d["z"],
            "rdrag_AP": rdrag_AP,
            "h_AP": h_AP,
            "nbinsmu": nbinsmu,
            "accboost": accboost,
            "Nlmax": Nlmax,
            "APst": APst,
        }
        return self

    def set_icc(
        self,
        Pshot: float,
        icc_fourier_file: str | None = None,
        icc_configspace_SN_file: str | None = None,
        icc_configspace_IC_file: str | None = None,
        Na: int | None = None,
        Nl: int | None = None,
        pmax: float = 0.3,
        accboost: int = 1,
        withmask: bool = True,
        windowk: float = 0.05,
        Nmax: int = 4096,
        bias: float = -2.1,
        window_param: float | None = 1,
        Nxmax: int = 4096,
        Nymax: int = 4096,
        xbias: float = -2.0,
        ybias: float = -2.0,
        windowxy_param: float | None = 1,
        inorder: bool = False,
    ):
        d = self.theory["eftpipe.eftlss"]["tracers"]["x"]
        d["with_icc"] = True
        d["icc"] = {
            "Pshot": Pshot,
            "icc_fourier_file": icc_fourier_file,
            "icc_configspace_SN_file": icc_configspace_SN_file,
            "icc_configspace_IC_file": icc_configspace_IC_file,
            "Na": Na,
            "Nl": Nl,
            "pmax": pmax,
            "accboost": accboost,
            "withmask": withmask,
            "windowk": windowk,
            "Nmax": Nmax,
            "bias": bias,
            "window_param": window_param,
            "Nxmax": Nxmax,
            "Nymax": Nymax,
            "xbias": xbias,
            "ybias": ybias,
            "windowxy_param": windowxy_param,
            "inorder": inorder,
        }
        return self

    def done(
        self, ellmax: Literal[2, 4] = 2, debug: bool = False, logging: bool = False
    ):
        if self._done:
            raise RuntimeError("already done")
        info: Any = {
            "theory": self.theory,
            "likelihood": self.likelihood,
            "params": self.params,
        }
        with do_nothing() if logging else disable_logging() as f:
            model = get_model(info, debug=debug)
            model.add_requirements(
                {
                    "nonlinear_Plk_interpolator": {
                        "x": {"ls": [*range(0, ellmax + 1, 2)]}
                    },
                    "Pk_interpolator": {
                        "nonlinear": False,
                        "z": [self.z],
                        "k_max": 5,
                        "vars_pairs": ("delta_nonu", "delta_nonu")
                        if self.use_cb
                        else ("delta_tot", "delta_tot"),
                    },
                }
            )
        self.model = model
        self._done = True
        return self

    def clone(self):
        ret = type(self)(1.0)
        ret.theory = deepcopy(self.theory)
        ret.likelihood = deepcopy(self.likelihood)
        ret.params = deepcopy(self.params)
        return ret

    def Plinear(self):
        if not self._done:
            raise RuntimeError("need to call done()")
        vars_pairs = (
            ("delta_nonu", "delta_nonu") if self.use_cb else ("delta_tot", "delta_tot")
        )
        fn = self.model.provider.get_Pk_interpolator(
            var_pair=vars_pairs, nonlinear=False, extrap_kmin=1e-6
        )
        h = self.params["H0"] / 100
        return lambda kh: fn.P(self.z, kh * h) * h**3

    def f(self):
        return self.model.theory[
            "eftpipe.classynu"
        ].classy.scale_independent_growth_factor_f(self.z)

    def Plk_mm(self):
        return self(
            b1A=1,
            c2A=math.sqrt(2) / 2,
            b3A=1,
            c4A=math.sqrt(2) / 2,
            cctA=0,
            cr1A=0,
            cr2A=0,
        )

    # fmt: off
    def __call__(
        self,
        b1A: float, c2A: float, b3A: float, c4A: float,
        cctA: float, cr1A: float, cr2A: float,
        ce0: float = 0.0, cemono: float = 0.0, cequad: float = 0.0,
        b1B: float = 0.0, c2B: float = 0.0, b3B: float = 0.0, c4B: float = 0.0,
        cctB: float = 0.0, cr1B: float = 0.0, cr2B: float = 0.0,
    ) -> PlkInterpolator:
        """
        Notes
        -----
        when computing the cross, ce0, cemono and cequad are used for x
        """
        if not self._done:
            raise RuntimeError("need to call done() first")
        if self.cross:
            sampled = dict(
                A_b1=b1A, A_c2=c2A, A_b3=b3A, A_c4=c4A,
                A_cct=cctA, A_cr1=cr1A, A_cr2=cr2A,
                B_b1=b1B, B_c2=c2B, B_b3=b3B, B_c4=c4B,
                B_cct=cctB, B_cr1=cr1B, B_cr2=cr2B,
                x_ce0=ce0, x_cemono=cemono, x_cequad=cequad,
            )
        else:
            sampled = dict(
                x_b1=b1A, x_c2=c2A, x_b3=b3A, x_c4=c4A,
                x_cct=cctA, x_cr1=cr1A, x_cr2=cr2A,
                x_ce0=ce0, x_cemono=cemono, x_cequad=cequad,
            )
        self.model.logpost(sampled)
        return self.model.provider.get_nonlinear_Plk_interpolator("x")
    # fmt: on
