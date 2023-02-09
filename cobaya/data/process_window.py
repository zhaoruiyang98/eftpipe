from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

GBWINDOW = Path("gbwindow")
WINDOW = Path("window")

norm = {
    "data.LRG.NGC": 6.178544514228104,
    "data.LRG.SGC": 3.0015458205664833,
    "data.ELG.NGC": 5.420129406342738,
    "data.ELG.SGC": 5.929721759936262,
    "noric.LRG.NGC": 6.196690731704751,
    "noric.LRG.SGC": 2.949184100567634,
    "noric.ELG.NGC": 5.724765399318868,
    "noric.ELG.SGC": 5.981649830039202,
}

alpha = {
    "data.LRG.NGC": 0.02043472833528684,
    "data.LRG.SGC": 0.021253667910211372,
    "data.ELG.NGC": 0.027598704461938163,
    "data.ELG.SGC": 0.02759870446193817,
    "noric.LRG.NGC": 0.05001692603563618,
    "noric.LRG.SGC": 0.050151088148061115,
    "noric.ELG.NGC": 0.049945259965807,
    "noric.ELG.SGC": 0.050006654928901224,
}


def smooth_window(
    win, spivot: float, window_length: int = 10, polyorder: int = 3, kind="interp"
):
    s = win[0]
    idx = np.searchsorted(s, spivot, side="right")
    swin = win.copy()
    for i in [3, 5, 7, 9, 11, 13, 15]:
        if kind == "interp":
            to_smooth = win[i, :idx][::-1]
            swin[i, :idx] = savgol_filter(to_smooth, window_length, polyorder)[::-1]
    return swin


def main():
    for kind in ["data", "noric"]:
        for tracer in ["LRG", "ELG"]:
            for cap in ["NGC", "SGC"]:
                key = f"{kind}.{tracer}.{cap}"
                suffix = "" if kind == "data" else "_complete"
                loadpath = GBWINDOW / f"{tracer}_{cap}{suffix}_QQ.dat"
                win = np.loadtxt(loadpath).T
                win = smooth_window(win, 20, 15)
                s = win[0, :].copy()
                win = win[[3, 5, 7, 9, 11, 13, 15], :].copy()
                mask = s >= 3.0
                s, win = s[mask], win[:, mask]
                win *= 1 / (8 * np.pi * norm[key]) * alpha[key] ** 2
                fn = interp1d(np.log(s), win, axis=-1, kind="cubic")
                ss = np.geomspace(s.min(), s.max(), 10000)
                win = fn(np.log(ss))
                Q0, Q2, Q4, Q6, Q8, Q10, Q12 = win
                plt.semilogx(ss, Q0, label="Q0")
                plt.semilogx(ss, Q2, label="Q2")
                plt.semilogx(ss, Q4, label="Q4")
                plt.semilogx(ss, Q6, label="Q6")
                plt.semilogx(ss, Q8, label="Q8")
                plt.semilogx(ss, Q10, label="Q10")
                plt.semilogx(ss, Q12, label="Q12")
                plt.legend(frameon=False)
                plt.xlim(ss.min(), ss.max())
                plt.ylim(-0.5, 1.0)
                plt.title(key)
                plt.show()
                output = np.vstack([ss, Q0, Q2, Q4, Q6, Q8, Q10, Q12]).T
                savepath = WINDOW / f"v3_{kind}_{tracer}_{cap}.dat"
                header = "s Q0 Q2 Q4 Q6 Q8 Q10 Q12"
                np.savetxt(savepath, output, header=header)


if __name__ == "__main__":
    main()
