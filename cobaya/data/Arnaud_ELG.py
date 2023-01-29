"""
files are located in eftpipe/cobaya/data/Arnaud_ELG_data
root: eftpipe/cobaya/
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def extract_covariance():
    path_template = "data/Arnaud_ELG_data/Covariance_ELGPk_{cap}_0.7z1.1_pix64_prerecon_0.6z1.1_postrecon.txt"
    names = ["catalogue_i", "catalogue_j", "ell_i", "ell_j", "k_i", "k_j", "Covij"]
    for cap in ["NGC", "SGC"]:
        df = pd.read_csv(
            path_template.format(cap=cap),
            comment="#",
            names=names,
            delim_whitespace=True,
        )
        df = df.query("catalogue_i == 'pre-recon' and catalogue_j == 'pre-recon'")
        cov1d = df["Covij"].to_numpy()
        nk = df["k_i"].unique().size
        nells = df["ell_i"].unique().size
        print(f"{cap=}, nk = {nk}, nells = {nells}")
        # raw data has already been sorted in standard order
        covarr = cov1d.reshape((nells, nells, nk, nk))
        cov = []
        for i in range(nells):
            for j in range(nk):
                flat = np.hstack([covarr[i, k, j, :] for k in range(nells)])
                cov.append(flat)
        cov = np.vstack(cov)
        np.savetxt(f"data/cov/pix64_cov_{cap}_E024.txt", cov)


def extract_power():
    path_template = "data/Arnaud_ELG_data/Data_ELGPk_{cap}_0.7z1.1_pix64_prerecon.txt"
    for cap in ["NGC", "SGC"]:
        path = path_template.format(cap=cap)
        with open(path) as f:
            header = f.readline()
            shotnoise = float(header.strip().split("=")[-1])
        data = np.loadtxt(path_template.format(cap=cap))
        k, P0, P2, P4 = data[:, 0], data[:, 1], data[:, 3], data[:, 5]
        output = np.vstack([k, P0, P2, P4]).T
        header = (" " * 5).join(["k", "P0", "P2", "P4"]) + f"\n{shotnoise=}"
        np.savetxt(f"data/DR16/Arnaud_pix64_ELG_{cap}.txt", output, header=header)


def extract_window():
    path_template = "data/Arnaud_ELG_data/Window_ELGPk_{cap}_0.7z1.1.txt"
    for cap in ["NGC", "SGC"]:
        path = path_template.format(cap=cap)
        with open(path) as f:
            header = f.readline().strip().expandtabs(4).replace("#", "")
        win = np.loadtxt(path)
        while win[0, 0] == 0:
            win = win[1:]
        s = win[:, 0]
        fn = interp1d(np.log(s), win[:, 1:].T, kind="cubic", axis=-1)
        sinterp = np.geomspace(s[0], s[-1], 10000)
        win = np.vstack([sinterp, fn(np.log(sinterp))]).T
        np.savetxt(f"data/window/Arnaud_ELG_{cap}.txt", win, header=header)


def main():
    extract_covariance()
    extract_power()
    extract_window()


if __name__ == "__main__":
    main()
