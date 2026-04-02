import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import metpy.calc as mpcalc
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units

# -----------------------------
# Read CSV safely as strings first
# -----------------------------
col_names = ['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']

with open('/Users/bri/Downloads/elreno_sounding.txt') as f:
    lines = f.readlines()[5:]  # skip first 5 header lines

data = []
for line in lines:
    # split by any whitespace, keep only the first 6 columns
    parts = line.strip().split()
    if len(parts) >= 6:
        data.append(parts[:6])

# Convert to DataFrame
df = pd.DataFrame(data, columns=col_names)

# Convert all to numeric, coerce errors to NaN
for col in col_names:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaNs in critical columns
df = df.dropna(subset=['pressure','temperature','dewpoint','direction','speed'])

# Sort descending pressure
df = df[df['pressure'] > 0].sort_values('pressure', ascending=False).reset_index(drop=True)

# -----------------------------
# Convert to 1D numpy arrays with units
# -----------------------------
p = df['pressure'].to_numpy(dtype=float) * units.hPa
T = df['temperature'].to_numpy(dtype=float) * units.degC
Td = df['dewpoint'].to_numpy(dtype=float) * units.degC
wind_speed = df['speed'].to_numpy(dtype=float) * units.knots
wind_dir = df['direction'].to_numpy(dtype=float) * units.degrees

# Wind components
u, v = mpcalc.wind_components(wind_speed, wind_dir)

# -----------------------------
# Debug shapes
# -----------------------------
print("Shapes check:", p.shape, T.shape, Td.shape, u.shape, v.shape)

# -----------------------------
# Plot Skew-T
# -----------------------------
fig = plt.figure(figsize=(9, 9))
add_metpy_logo(fig, 115, 100)
skew = SkewT(fig, rotation=45)

skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')
skew.plot_barbs(p, u, v)

skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-40, 60)
skew.ax.set_xlabel(f'Temperature ({T.units:~P})')
skew.ax.set_ylabel(f'Pressure ({p.units:~P})')

# LCL and parcel profile
lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
skew.plot(p, prof, 'k', linewidth=2)

# Shade CIN and CAPE
skew.shade_cin(p, T, prof, Td)
skew.shade_cape(p, T, prof)

# 0°C isotherm
skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

# Special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

plt.show()
