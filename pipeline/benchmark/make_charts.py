"""Generate chart-reading items with exact answers in benchmark/data/charts.jsonl and charts/*.png.

Five chart types that recur in materials papers, four items each. The tolerance of a read
is 3% of the axis span on linear axes, 12% on log axes and 1% where the value is printed,
stored relative to the answer.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(__file__).parent / "data"
(out / "charts").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
items = []


def save(fig, question, answer, unit, tolerance):
    name = f"chart-{len(items) + 1:02d}"
    fig.tight_layout()
    fig.savefig(out / "charts" / f"{name}.png", dpi=150)
    plt.close(fig)
    items.append({"id": name, "image": f"charts/{name}.png", "question": question,
                  "answer": round(float(answer), 3), "unit": unit, "tolerance": round(float(tolerance), 4)})


def span_tolerance(ax, answer):
    lo, hi = ax.get_ylim()
    return 0.03 * (hi - lo) / abs(answer)


labels = ["Monolithic", "Laminated", "Brick-and-mortar", "Freeze-cast", "Graded", "Bouligand"]

for i in range(4):  # stress-strain curves, read a peak stress
    fig, ax = plt.subplots(figsize=(6, 4))
    names = rng.choice(labels, 3, replace=False)
    peaks = rng.uniform(80, 480, 3).round(0)
    for name, peak in zip(names, peaks):
        e_peak = peak / rng.uniform(250e3, 450e3)  # stress over a modulus of 250 to 450 GPa, in MPa
        strain = np.linspace(0, e_peak, 50)
        tail = np.linspace(e_peak, e_peak * rng.uniform(1.05, 2.0), 30)
        stress = np.concatenate([peak * strain / e_peak, peak * np.exp(-3 * (tail - e_peak) / e_peak)])
        ax.plot(np.concatenate([strain, tail]) * 100, stress, label=name)
    ax.set(xlabel="Strain (%)", ylabel="Flexural stress (MPa)", ylim=(0, None))
    ax.legend(frameon=False)
    k = rng.integers(3)
    save(fig, f"What is the maximum flexural stress of the {names[k]} sample?", peaks[k], "MPa",
         span_tolerance(ax, peaks[k]))

for i in range(4):  # bars with error bars, half with printed values
    fig, ax = plt.subplots(figsize=(6, 4))
    names = rng.choice(labels, 5, replace=False)
    values = rng.uniform(2.5, 18, 5).round(1)
    bars = ax.bar(names, values, yerr=values * 0.08, capsize=4, color="0.6", edgecolor="k")
    if i % 2:
        ax.bar_label(bars, fmt="%.1f", padding=6)
        ax.margins(y=0.15)
    ax.set(ylabel="Fracture toughness (MPa m$^{1/2}$)")
    ax.tick_params(axis="x", labelrotation=20)
    k = rng.integers(5)
    tol = 0.01 if i % 2 else span_tolerance(ax, values[k])
    save(fig, f"What is the fracture toughness of the {names[k]} sample?", values[k], "MPa m^1/2", tol)

for i in range(4):  # property map with a log axis, read one labeled point
    fig, ax = plt.subplots(figsize=(6, 4.5))
    samples = [f"Sample {c}" for c in "ABCDEFGH"]  # neutral labels, so prior knowledge cannot help
    density = rng.uniform(1.0, 6.0, 8).round(2)
    modulus = 10 ** rng.uniform(0, 2.7, 8)
    ax.semilogy(density, modulus, "o", color="k")
    for m, x, y in zip(samples, density, modulus):
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set(xlabel="Density (g/cm$^3$)", ylabel="Young's modulus (GPa)")
    ax.grid(True, which="both", lw=0.3)
    k = rng.integers(8)
    save(fig, f"What is the Young's modulus of {samples[k]}?", modulus[k], "GPa", 0.12)

for i in range(4):  # traction-separation laws, read a peak traction or a final opening
    fig, ax = plt.subplots(figsize=(6, 4))
    peak, d0, df = rng.uniform(20, 300), rng.uniform(0.5, 2), rng.uniform(5, 20)
    d = np.linspace(0, df * 1.15, 400)
    t = np.where(d < d0, peak * d / d0, np.clip(peak * (df - d) / (df - d0), 0, None))
    ax.plot(d, t, color="k")
    ax.set(xlabel="Separation (µm)", ylabel="Traction (MPa)", ylim=(0, None), xlim=(0, None))
    if i % 2:
        save(fig, "What is the peak traction of this cohesive law?", round(peak), "MPa", span_tolerance(ax, peak))
    else:
        lo, hi = ax.get_xlim()
        save(fig, "At what separation does the traction first return to zero?", round(df, 1), "um",
             0.03 * (hi - lo) / df)

for i in range(4):  # rising R-curves, read toughness at a crack extension
    fig, ax = plt.subplots(figsize=(6, 4))
    names = rng.choice(labels, 2, replace=False)
    a = np.linspace(0, 500, 100)
    params = [(rng.uniform(2, 5), rng.uniform(2, 12), rng.uniform(80, 250)) for _ in names]
    for name, (k0, gain, scale) in zip(names, params):
        ax.plot(a, k0 + gain * (1 - np.exp(-a / scale)), marker="o", markevery=10, ms=4, label=name)
    ax.set(xlabel="Crack extension (µm)", ylabel="K$_R$ (MPa m$^{1/2}$)", ylim=(0, None))
    ax.legend(frameon=False)
    ax.grid(True, lw=0.3)
    k, at = rng.integers(2), [100, 200, 300, 400][i]
    k0, gain, scale = params[k]
    value = k0 + gain * (1 - np.exp(-at / scale))
    save(fig, f"What is the crack resistance K_R of the {names[k]} sample at {at} µm of crack extension?",
         value, "MPa m^1/2", span_tolerance(ax, value))

(out / "charts.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
print(len(items), "items")
