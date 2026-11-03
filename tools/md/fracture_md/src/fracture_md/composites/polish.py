"""Polish (gloss) retention after simulated toothbrushing - CALIBRATED EMPIRICAL model.

Target metric: 3M technical data sheet for Filtek Supreme Flowable (2021, "3M internal data"): tiles wet-polished,
24 h water at 37 C, 60-degree gloss, automatic toothbrush machine with toothpaste, 6000 strokes; the value is the
gloss retained (% of initial).  Filtek Supreme Flowable 66 %, average of six flowables 43.1 % (anchors).

Mechanism (3M TDS, Mitra et al. JADA 2003): when the matrix is abraded, a discrete particle is plucked whole and
leaves a pit of its own size; a nanocluster (20 nm SiO2 + 4-11 nm ZrO2) wears away primary particle by primary
particle.  The model therefore uses the WEAR-UNIT size d_w of each filler (primary-particle size for clusters,
particle size for discrete fillers) and the volume-weighted geometric mean d_eff = exp(sum v_i ln d_w,i):

    GR [%] = a - b log10(d_eff / 1 um)          (clipped to 0-100)

a, b are fixed by the TWO anchors (Filtek-proxy and flowable-average-proxy compositions from direct_design) -
a two-point calibration, NOT a regression: it has zero residual by construction and no independent validation.
Prediction interval: +/- 15 percentage points (ASSUMED).  Rationale: the literature is inconsistent about any
nanofill advantage in gloss retention (Amaya-Pajares et al. 2022 review: no consistent in vitro evidence favouring
nano/submicron over microhybrids), so particle size explains only part of the product-to-product scatter; matrix
wear resistance (DC, filler-matrix coupling), polishing protocol and toothpaste RDA are not modelled.
Short fibres (e.g. everX Flow, ~6 um x 140 um-scale E-glass) are outside the calibrated domain; if present
they are treated as wear units of their diameter and the result is flagged out-of-domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from fracture_md.schema.common import CheckResult, Citation
from fracture_md.units import Unit

CITES = {
    "3M_TDS": Citation(key="3M_TDS", text="3M Filtek Supreme Flowable Restorative Technical Data Sheet (2021): polish retention after 6000 toothbrush strokes, 60 deg gloss; Filtek 66 %, flowable average 43.1 % (3M internal data)",
                       url="https://multimedia.3m.com/mws/media/2084833O/3m-filtek-supreme-flowable-restorative-technical-data-sheet.pdf"),
    "Mitra2003": Citation(key="Mitra2003", text="Mitra SB, Wu D, Holmes BN. An application of nanotechnology in advanced dental materials. J Am Dent Assoc 2003;134:1382-1390",
                          doi="10.14219/jada.archive.2003.0054"),
    "AmayaPajares2022": Citation(key="AmayaPajares2022", text="Amaya-Pajares SP et al. Development and maintenance of surface gloss of dental composites after polishing and brushing: review of the literature. J Esthet Restor Dent 2022;34:15-41",
                                 doi="10.1111/jerd.12875"),
}

# Wear-unit size (um) per filler (harness constituent names / aliases).  Sources: 3M TDS (cluster primaries
# 20 nm SiO2 / 4-11 nm ZrO2; discrete 20 and 75 nm silica; YbF3 0.1-5.0 um -> 0.7 um geometric mid ASSUMED),
# constituents.json d50 for the Ba glasses and fumed silica.
WEAR_UNIT_UM = {
    "nanocluster": 0.02, "zirconia-silica nanocluster filler": 0.02,
    "fumed silica": 0.04, "nano zirconia": 0.02,
    "ba glass 0.4": 0.4, "ba glass 0.7": 0.7, "ba glass 1.5": 1.5, "spherical silica 0.5": 0.5, "zirconia 0.4": 0.4,
    "ybf3": 0.7,
    "short glass fibre": 6.0,  # diameter; OUT OF DOMAIN
}
OUT_OF_DOMAIN = {"short glass fibre"}
PI_PP = 15.0  # +/- percentage points (ASSUMED)


def d_eff_um(filler_vol_shares: dict[str, float]) -> float:
    s = sum(filler_vol_shares.values())
    return math.exp(sum(v / s * math.log(WEAR_UNIT_UM[k.lower()]) for k, v in filler_vol_shares.items() if v > 0))


@dataclass
class GlossCalibration:
    a: float
    b: float
    anchors: dict = field(default_factory=dict)


def calibrate(anchor_fillers: dict[str, tuple[dict, float]]) -> GlossCalibration:
    """Two-point calibration: {label: (filler volume shares, retained gloss %)}."""
    (l1, (f1, g1)), (l2, (f2, g2)) = list(anchor_fillers.items())[:2]
    x1, x2 = math.log10(d_eff_um(f1)), math.log10(d_eff_um(f2))
    if abs(x1 - x2) < 1e-6:
        raise ValueError("anchors have the same effective wear-unit size")
    b = -(g1 - g2) / (x1 - x2)
    a = g1 + b * x1
    return GlossCalibration(a, b, {l1: {"d_eff_um": 10 ** x1, "gloss": g1}, l2: {"d_eff_um": 10 ** x2, "gloss": g2}})


@dataclass
class GlossPrediction:
    value: float
    low: float
    high: float
    d_eff_um: float
    in_domain: bool
    unit: str = "% gloss retained (6000 strokes, 60 deg)"
    method: str = "calibrated empirical: GR = a - b log10(d_eff); two-anchor calibration (3M TDS); +/-15 pp ASSUMED"


def predict_gloss_retention(filler_vol_shares: dict[str, float], cal: GlossCalibration) -> GlossPrediction:
    d = d_eff_um(filler_vol_shares)
    g = cal.a - cal.b * math.log10(d)
    clip = lambda x: min(max(x, 0.0), 100.0)
    ind = not any(k.lower() in OUT_OF_DOMAIN for k, v in filler_vol_shares.items() if v > 0)
    # domain in d_eff: between the anchors extended by a factor 3 either side
    ds = [v["d_eff_um"] for v in cal.anchors.values()]
    ind = ind and (min(ds) / 3.0 <= d <= max(ds) * 3.0)
    return GlossPrediction(clip(g), clip(g - PI_PP), clip(g + PI_PP), d, ind)


def verification_checks(cal: GlossCalibration) -> list[CheckResult]:
    out = []
    for lab, v in cal.anchors.items():
        g = cal.a - cal.b * math.log10(v["d_eff_um"])
        out.append(CheckResult(name=f"gloss_anchor_{lab}", passed=abs(g - v["gloss"]) < 1e-9, metric=g, tolerance=v["gloss"], unit=Unit.PERCENT,
                               details=f"{lab}: {g:.2f} vs {v['gloss']} % (exact by construction - calibration identity, not validation)"))
    out.append(CheckResult(name="gloss_monotonic_in_size", passed=cal.b > 0, metric=cal.b, tolerance=0.0,
                           details=f"b = {cal.b:.2f} pp per decade > 0: finer wear units retain more gloss (Mitra 2003 mechanism)"))
    return out
