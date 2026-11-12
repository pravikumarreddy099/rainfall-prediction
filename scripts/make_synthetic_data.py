"""Generate a SYNTHETIC stand-in with the same schema as the real Rainfall CSV.

For smoke-testing the pipeline only. Never report metrics from this file.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 366
temp = rng.normal(24, 5, n)
dew = temp - rng.gamma(2.0, 3.0, n)
humidity = np.clip(rng.normal(80, 10, n), 36, 98).round()
cloud = np.clip(rng.normal(71, 22, n), 0, 100).round()
sunshine = np.clip(12 - cloud / 10 + rng.normal(0, 2.5, n), 0, 12.1).round(1)
pressure = rng.normal(1013, 6, n).round(1)
logit = 0.04 * (humidity - 80) + 0.03 * (cloud - 70) - 0.15 * (sunshine - 4) \
    - 0.05 * (pressure - 1013) + 0.9
rain = rng.random(n) < 1 / (1 + np.exp(-logit))

df = pd.DataFrame({
    "day": np.tile(np.arange(1, 32), 12)[:n],
    "pressure ": pressure,
    "maxtemp": (temp + rng.uniform(1, 5, n)).round(1),
    "temparature": temp.round(1),
    "mintemp": (temp - rng.uniform(1, 5, n)).round(1),
    "dewpoint": dew.round(1),
    "humidity ": humidity,
    "cloud ": cloud,
    "rainfall": np.where(rain, "yes", "no"),
    "sunshine": sunshine,
    " winddirection": rng.choice(np.arange(10, 360, 10), n).astype(float),
    "windspeed": rng.gamma(4, 5, n).round(1),
})
df.loc[10, " winddirection"] = np.nan
df.loc[20, "windspeed"] = np.nan

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_sample.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"wrote {out} ({len(df)} rows, rain rate {rain.mean():.0%})")
