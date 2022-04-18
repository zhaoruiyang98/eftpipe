import math
import os
from typing import List
from scipy.optimize import fsolve
from cobaya.log import LoggedError
from cobaya.theories.classy import classy


def get_masses(delta_m, sum_masses, hierarchy):
    if hierarchy == "normal":
        delta_m21_square = delta_m["delta_m21_square"]
        delta_m31_square = delta_m["delta_m31_square"]
        func = lambda x: (
            -sum_masses + x + math.sqrt(delta_m21_square + x ** 2)
            + math.sqrt(delta_m31_square + x ** 2)
        )
        m0, _, success, _ = fsolve(func, 0., full_output=True)
        m0 = m0[0]
        out = [
            m0,
            math.sqrt(delta_m21_square + m0 ** 2),
            math.sqrt(delta_m31_square + m0 ** 2),
        ]
    else:
        delta_m21_square = delta_m["delta_m21_square"]
        delta_m32_square = math.fabs(delta_m["delta_m32_square"])
        func = lambda x: (
            -sum_masses + x + math.sqrt(delta_m32_square + x ** 2)
            + math.sqrt(delta_m32_square - delta_m21_square + x ** 2)
        )
        m0, _, success, _ = fsolve(func, 0., full_output=True)
        m0 = m0[0]
        out = [
            m0,
            math.sqrt(delta_m32_square + m0 ** 2),
            math.sqrt(delta_m32_square - delta_m21_square + m0 ** 2),
        ]
    return out, success


def get_delta_m(extra_args, hierarchy):
    """construct delta_m dictionary

    default values are taken from 1907.12598
    """
    if hierarchy == "degenerate":
        return {}
    # (3, 2), (2, 1), (1, 3)
    if hierarchy == "normal":
        default = [-1 * (7.39e-5 - 2.525e-3), 7.39e-5, -2.525e-3]
    else:
        default = [-2.512e-3, 7.39e-5, -1 * (-2.512e-3 + 7.39e-5)]

    inputs: List[float] = [None, None, None]  # type: ignore
    inputs[0] = extra_args.pop('delta_m32_square', None)
    inputs[1] = extra_args.pop('delta_m21_square', None)
    x = extra_args.pop('delta_m31_square', None)
    inputs[2] = x if x is None else -x

    flag = sum(0 if x is None else 1 for x in inputs)
    if flag == 3:
        if sum(inputs) != 0:
            raise ValueError("invalid inputs")
    elif flag == 2:
        for i, v in enumerate(inputs):
            if v is None:
                ileft = (i - 1) % 3
                iright = (i + 1) % 3
                inputs[i] = -(inputs[ileft] + inputs[iright])
    elif flag == 1:
        raise ValueError("Please specify at least two delta_m_square")
    else:
        inputs = default

    out = {
        "delta_m32_square": inputs[0],
        "delta_m21_square": inputs[1],
        "delta_m31_square": -inputs[2],
    }
    return out


class classynu(classy):
    """enhanced classy theory with neutrino hierarchy supports
    """

    def initialize(self):
        super().initialize()
        hierarchy = self.extra_args.pop('neutrino_hierarchy', 'degenerate')
        if hierarchy not in ('degenerate', 'normal', 'inverted'):
            raise LoggedError(
                self.log,
                "neutrino hierarchy must be "
                "'degenerate', 'normal' or 'inverted'"
            )
        self._hieararchy = hierarchy
        N_ur = self.extra_args.pop('N_ur', None)
        N_ncdm = self.extra_args.pop('N_ncdm', None)
        # if you want to use N_ur to get N_eff equal to 3.046 in the early universe,
        # then you should pass here respectively 2.0328,1.0196,0.00641
        # Thus the following only works for N_eff=3.046
        if hierarchy in ("normal", "inverted"):
            if (N_ur is not None) or (N_ncdm is not None):
                self.mpi_info(
                    "N_ur and N_ncdm settings are overrided for "
                    "neutrino hierarchy '%s'",
                    hierarchy,
                )
            N_ur = 0.00641
            N_ncdm = 3
        else:
            if N_ur is None or N_ncdm is None:
                N_ur = 2.0328
                N_ncdm = 1
                self.mpi_warning(
                    "default settings N_ncdm=1, N_ur=2.0328, "
                    "Planck18 baseline is used",
                )
        self._N_ur = N_ur
        self._N_ncdm = N_ncdm
        self._delta_m = get_delta_m(self.extra_args, hierarchy)
        self.log.debug("difference of neutrino masses: %s", self._delta_m)

    def initialize_with_params(self):
        super().initialize_with_params()
        input_params = self.input_params
        if 'mnu' in input_params and 'm_ncdm' in input_params:
            raise LoggedError(
                self.log, "mnu and m_ncdm cannot appear at the same time")
        if self._hieararchy != 'degenerate' and 'm_ncdm' in input_params:
            raise LoggedError(
                self.log, "m_ncdm is only supported for degenerate hierarchy")

    # override
    def set(self, params_values_dict):
        # If no output requested, remove arguments that produce an error
        # (e.g. complaints if halofit requested but no Cl's computed.)
        # Needed for facilitating post-processing
        if not self.extra_args["output"]:
            for k in ["non linear"]:
                self.extra_args.pop(k, None)
        # Prepare parameters to be passed: this-iteration + extra
        args = {self.translate_param(
            p): v for p, v in params_values_dict.items()}
        args.update(self.extra_args)
        self._update_args_dict_with_neutrino(args)
        # Generate and save
        self.log.debug("Setting parameters: %r", args)
        self.classy.set(**args)

    def _update_args_dict_with_neutrino(self, args_dict):
        """update args_dict with neutrino settings

        Notes
        -----
        Although some checks are performed in initialize and initialize_with_params, 
        mnu and m_ncdm may still appear at the same time.
        """
        args_dict["N_ur"] = self._N_ur
        args_dict["N_ncdm"] = self._N_ncdm
        if self._hieararchy == "degenerate":
            mnu = args_dict.pop('mnu', None)
            if mnu is not None:
                m_ncdm = [mnu / self._N_ncdm for _ in range(self._N_ncdm)]
                args_dict["m_ncdm"] = ','.join(map(str, m_ncdm))
        else:
            mnu = args_dict['mnu']
            args_dict.pop('mnu')
            (m1, m2, m3), success = get_masses(
                self._delta_m, mnu, self._hieararchy)
            if not success:
                raise LoggedError(self.log, "failed to solve m_ncdm")
            args_dict["m_ncdm"] = ','.join(map(str, [m1, m2, m3]))


    # override: import classy instead of classynu
    @classmethod
    def get_path(cls, path):
        return os.path.realpath(os.path.join(path, "code", "classy"))
