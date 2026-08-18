"""
plot_gaa_symmetry.py
---------------------
Parse Sentaurus TCAD electrical-characteristic .plt files (DF-ISE text
format) and plot them, to help check symmetry of n-type / p-type GAAFETs.
 
YOUR FILES ARE "DF-ISE text" FORMAT, e.g.:
 
    DF-ISE text
 
    Info {
      version   = 1.0
      type      = xyplot
      datasets  = [
        "time"
        "substrate OuterVoltage" "substrate InnerVoltage" ... ]
      functions = [ ... ]
    }
 
    Data {
          0.00000000000000E+00
          0.00000000000000E+00   0.00000000000000E+00   ...
          ...
    }
 
This is NOT Tecplot format (that's a different, older Sentaurus output
style) -- this parser is written specifically for DF-ISE.
 
Each row of data = one bias/time point. The columns are exactly the names
listed in "datasets": time, then 8 quantities per electrode (OuterVoltage,
InnerVoltage, QuasiFermiPotential, DisplacementCurrent, eCurrent, hCurrent,
TotalCurrent, Charge) for however many electrodes your device has
(commonly: substrate, gate, source, drain).
"""
 
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
 
# ----------------------------------------------------------------------
# STEP 1: Parse a DF-ISE .plt file into a pandas DataFrame
# ----------------------------------------------------------------------
def parse_dfise(filepath):
    """
    Reads a Sentaurus DF-ISE .plt file and returns a DataFrame where
    each column is one of the named datasets (e.g. "gate OuterVoltage",
    "drain TotalCurrent") and each row is one bias point.
    """
    with open(filepath, "r", errors="ignore") as f:
        text = f.read()
 
    ds_match = re.search(r'datasets\s*=\s*\[(.*?)\]', text, re.DOTALL)
    if not ds_match:
        raise ValueError(f"Could not find 'datasets' block in {filepath} "
                          f"-- is this really a DF-ISE .plt file?")
    names = re.findall(r'"([^"]+)"', ds_match.group(1))
 
    data_match = re.search(r'Data\s*\{(.*)\}', text, re.DOTALL)
    if not data_match:
        raise ValueError(f"Could not find 'Data' block in {filepath}")
    nums = [float(x) for x in data_match.group(1).split()]
 
    ncol = len(names)
    nrow = len(nums) // ncol
    if nrow * ncol != len(nums):
        print(f"Warning: {filepath} data length ({len(nums)}) is not an "
              f"exact multiple of the number of columns ({ncol}); "
              f"truncating trailing partial row.")
 
    arr = np.array(nums[: nrow * ncol]).reshape(nrow, ncol)
    return pd.DataFrame(arr, columns=names)
 
 
# ----------------------------------------------------------------------
# STEP 2: Auto-detect which electrode was swept in a given file
# ----------------------------------------------------------------------
def detect_swept_electrode(df, electrodes=("gate", "drain", "source", "substrate")):
    """
    Looks at each electrode's OuterVoltage column and returns the name of
    whichever one actually changes value across the sweep (the others are
    normally held fixed as bias conditions).
    """
    swept = []
    for e in electrodes:
        col = f"{e} OuterVoltage"
        if col in df.columns and df[col].nunique() > 1:
            swept.append(e)
    return swept
 
 
# ----------------------------------------------------------------------
# STEP 3: Plot helper
# ----------------------------------------------------------------------
def plot_iv(df, x_col, y_col, label=None, log_y=False, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    y = np.abs(df[y_col]) if log_y else df[y_col]
    ax.plot(df[x_col], y, 'o-', markersize=4, label=label, **kwargs)
    if log_y:
        ax.set_yscale('log')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, which='both', alpha=0.3)
    return ax
 
 
def check_symmetry(df_normal, df_swapped, x_col, y_col):
    """
    Compares a 'normal' sweep to a 'source/drain swapped' sweep of the
    SAME device on a common x-grid and reports the max % deviation.
    Use this once you have two files for the same device: one with
    source/drain wired normally, one with them swapped in the sdevice
    command file.
    """
    common_x = np.linspace(
        max(df_normal[x_col].min(), df_swapped[x_col].min()),
        min(df_normal[x_col].max(), df_swapped[x_col].max()),
        200,
    )
    y_normal = np.interp(common_x, df_normal[x_col], np.abs(df_normal[y_col]))
    y_swapped = np.interp(common_x, df_swapped[x_col], np.abs(df_swapped[y_col]))
    max_dev_pct = np.max(np.abs(y_normal - y_swapped) / (y_normal + 1e-30)) * 100
    return max_dev_pct
 
 
# ----------------------------------------------------------------------
# DEMO: run on your two uploaded files
# ----------------------------------------------------------------------
if __name__ == "__main__":
 
    df_n70 = parse_dfise("OutputCharacteristics_pGAAFET.plt")
    df_n614 = parse_dfise("OutputCharacteristics_nGAAFET.plt")
 
    print("IdVg_n70_des.plt  -> swept electrode(s):", detect_swept_electrode(df_n70))
    print("OutputCharacteristics_n614_des.plt -> swept electrode(s):", detect_swept_electrode(df_n614))
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
 
    plot_iv(df_n70, "drain OuterVoltage", "drain TotalCurrent",
            label="Vg = -0.5 V", ax=axes[0])
    axes[0].set_title("IdVg_n70_des.plt\n(actually an Id-Vd sweep at fixed Vg)")
    axes[0].legend()
 
    plot_iv(df_n614, "drain OuterVoltage", "drain TotalCurrent",
            label="Vg = +0.5 V", ax=axes[1])
    axes[1].set_title("OutputCharacteristics_n614_des.plt\n( sweep at fixed Vg)")
    axes[1].legend()
 
    fig.tight_layout()
    fig.savefig("demo_plots.png", dpi=200)
    print("Saved demo_plots.png")
 
    # NOTE: these two files are DIFFERENT devices (n70 vs n614), not a
    # normal/swapped pair of the same device, so check_symmetry() isn't
    # meaningful between them. Once you have a matched pair (same device,
    # source/drain roles swapped), use:
    #
    #   dev = check_symmetry(df_normal, df_swapped,
    #                         "drain OuterVoltage", "drain TotalCurrent")
    #   print(f"Max relative deviation: {dev:.2f}%")
 