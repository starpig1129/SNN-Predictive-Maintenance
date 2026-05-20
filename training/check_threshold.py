import pandas as pd
import numpy as np
import pathlib

def delta_rate(path, thresh):
    sig = pd.read_csv(path)["accel_z"].to_numpy()
    spikes = np.abs(np.diff(sig)) > thresh
    return spikes.mean()

for f in ["normal.csv", "idle.csv", "anomaly.csv"]:
    p = pathlib.Path("data") / f
    print(f"\n{f}")
    for t in [0.10, 0.5, 1.0, 2.0, 3.0, 5.0]:
        print(f"  thresh={t:.2f}  spike_rate={delta_rate(p, t):.3f}")
