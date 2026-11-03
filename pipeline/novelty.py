"""How far the hypothesis of a design and its toolpath pattern lies from the corpus, as a novelty score in [0, 1].

The hypothesis is the set of categorical choices of a design, such as nacre, alumina, graphite,
direct ink writing and crack deflection, and the structural features its pattern uses, such as tablets
or helicoidal plies. Combination novelty asks how many of these terms one record of the corpus already
names together. A record is a passage, a figure caption or a knowledge graph triple. Semantic novelty
asks how close the nearest text passage comes to the hypothesis in one sentence, ranked among the same
sentences for random hypotheses and patterns, so that a short sentence is compared with short sentences.
The score is the mean of the two.
proposals() runs the other way. It finds hypotheses that the knowledge graph supports one link at a time and that no
record names whole, for the planner to offer the designer.
What depends only on the corpus is computed at the first call and kept in corpus["novelty"], in memory.
"""

import itertools
import re

import numpy as np

from tools.dft import dft_cleavage
from tools.md import md_interface

from .design import DESIGN
from .pattern import PATTERN, complete, describe

# The categorical fields of a design, in the order the statement names them.
FIELDS = ["biological_model", "ceramic", "interlayer", "process", "mechanism"]
# The biological models with spellings below. The design names its model in free text.
MODELS = DESIGN["biological_model"].get("enum") or DESIGN["biological_model"]["examples"]
# The spellings a paper may use for each option and pattern feature. They match case-insensitively as whole
# words, with a space or a hyphen between words and an optional plural ending. A missed spelling makes a
# design look novel, and the search would then favor it, so the lists lean broad, as glass for silica glass.
ALIASES = {
    "nacre": ["nacre", "nacreous", "mother of pearl", "brick and mortar", "abalone"],
    "conch shell": ["conch shell", "conch", "Strombus gigas", "crossed lamellar"],
    "bone": ["bone", "osteon", "osteonal", "Haversian"],
    "Bouligand": ["Bouligand", "helicoid", "helicoidal", "twisted plywood", "mantis shrimp", "dactyl club"],
    "tooth enamel": ["tooth enamel", "dental enamel", "dentin", "dentine", "dentin-enamel junction"],
    "fish scale": ["fish scale", "Arapaima", "ganoid scale", "elasmoid scale"],
    "crack deflection": ["crack deflection", "crack deflect", "cracks deflect", "deflected crack", "deflect cracks",
                         "deflects cracks", "deflect the crack", "deflects the crack", "deflection of the crack",
                         "crack kinking"],
    "crack bridging": ["crack bridging", "bridging", "crack bridge", "bridged crack", "bridges the crack"],
    "compressive residual stress": ["compressive residual stress", "residual compressive stress", "residual stress",
                                    "prestressed", "pre-stressed"],
    "alumina": ["alumina", "Al2O3", "Al_2O_3", "Al$_2$O$_3$", "aluminium oxide", "aluminum oxide", "corundum", "sapphire"],
    "silicon carbide": ["silicon carbide", "SiC"],
    "silica glass": ["silica glass", "fused silica", "silica", "SiO2", "SiO_2", "glass", "borosilicate"],
    "graphite": ["graphite", "graphitic"],
    "epoxy": ["epoxy", "epoxies", "epoxide"],
    "PMMA": ["PMMA", "poly(methyl methacrylate)", "polymethyl methacrylate", "polymethylmethacrylate"],
    "direct ink writing": ["direct ink writing", "direct ink write", "direct writing", "DIW", "robocasting", "robocast",
                           "robocasted", "extrusion-based", "material extrusion", "paste extrusion"],
    "vat photopolymerization": ["vat photopolymerization", "photopolymerization", "photopolymerisation",
                                "stereolithography", "stereolithographic", "SLA", "DLP", "digital light processing"],
    "binder jetting": ["binder jetting", "binder jet", "binder jetted"],
    "tablets": ["tablet", "platelet", "brick", "brick and mortar", "staggered"],
    "helicoidal plies": ["Bouligand", "helicoid", "helicoidal", "twisted plywood", "plywood", "rotated plies",
                         "rotated layers", "ply rotation"],
    "crossed lamellae": ["crossed lamellar", "crossed lamellae", "crossed lamella", "cross lamellar", "cross ply",
                         "cross plies", "crossed plies", "angle ply", "angle plies"],
    "concentric rings": ["concentric", "osteon", "osteonal", "Haversian", "annular"],
    "wavy interfaces": ["wavy", "waviness", "undulating", "undulation", "sinusoidal", "corrugated", "suture", "sutured",
                        "interlocking"],
    "graded modulus": ["graded", "gradient", "functionally graded", "FGM"],
    "core-shell roads": ["core-shell", "coextrusion", "coextruded", "co-extrusion", "co-extruded", "coaxial",
                         "fibrous monolith"],
}


def features(pattern):
    """The structural features a pattern uses, as hypothesis terms."""
    p = complete(pattern)
    return [term for term, used in [
        ("tablets", p["break_length_h"]),
        ("helicoidal plies", p["ply_rotation_deg"] and not p["ply_rotation_alternates"]),
        ("crossed lamellae", p["ply_rotation_deg"] and p["ply_rotation_alternates"]),
        ("concentric rings", p["road_path"] == "rings"),
        ("wavy interfaces", p["road_path"] == "waves" or p["ply_wave_amplitude_h"]),
        ("graded modulus", p["modulus_profile"] != "uniform"),
        ("core-shell roads", p["interlayer_placement"] == "shell")] if used]


# The pattern parameters that carry each feature, as the architect is told them.
CARRIES = {"tablets": "break_length_h", "helicoidal plies": "ply_rotation_deg, with ply_rotation_alternates false",
           "crossed lamellae": "ply_rotation_deg, with ply_rotation_alternates true", "concentric rings": "road_path rings and ring_spacing_w",
           "wavy interfaces": "road_path waves, or ply_wave_amplitude_h and ply_wave_length_h",
           "graded modulus": "modulus_profile and outer_layer_modulus_GPa", "core-shell roads": "interlayer_placement shell"}


def regex(spelling):
    """A spelling as a regular expression for lowercase text, with a space or a hyphen between words and an optional plural."""
    return re.escape(spelling.lower()).replace(r"\ ", r"[\s-]+").replace(r"\-", r"[\s-]+") + r"(?:e?s)?(?!\w)"


def spellings(term):
    """The spellings of a term. Free text, as a biological model, also takes the spellings of every listed model it
    names, so that "abalone nacre" counts every passage on nacre."""
    if term in ALIASES:
        return ALIASES[term]
    return [term] + [a for m in MODELS if any(re.search(r"(?<!\w)" + regex(b), term.lower()) for b in ALIASES[m]) for a in ALIASES[m]]


def statement(design, pattern):
    """The hypothesis in one sentence, such as "An alumina laminate inspired by nacre, with graphite interlayers, made by
    direct ink writing, toughened by crack deflection, printed with straight roads, interlayer sheets between plies." """
    d = design
    s = f"{d.get('ceramic', 'ceramic')} laminate" + (f" inspired by {d['biological_model']}," if "biological_model" in d else "")
    s += f" with {d['interlayer']} interlayers," if "interlayer" in d else ""
    s += f" made by {d['process']}," if "process" in d else ""
    s += f" toughened by {d['mechanism']}," if "mechanism" in d else ""
    s += f" printed with {describe(pattern)}"
    return ("An " if s[0] in "aeiouAEIOU" else "A ") + s + "."


def random_pattern(rng):
    """A pattern whose parameters each keep their default or, with probability 1/2, take a random option or value."""
    return {k: spec["options"][rng.integers(len(spec["options"]))] if "options" in spec else round(float(rng.uniform(*spec["range"])), 1)
            for k, spec in PATTERN.items() if rng.random() < 0.5 or spec["default"] is None}


def records(corpus):
    """The records of the corpus, passages and captions then triples, in one lowercase string."""
    cache = corpus.setdefault("novelty", {})
    if "text" not in cache:
        texts = [c["text"].lower() for c in corpus["chunks"]]
        texts += [f"{t['head']} {t['relation']} {t['tail']}".lower() for t in corpus["triples"]]
        cache["ids"] = [c["id"] for c in corpus["chunks"]] + [f"kg:{i}" for i in range(len(corpus["triples"]))]
        cache["triple"] = np.arange(len(texts)) >= len(corpus["chunks"])
        # All records in one string, so that one regex scan finds a spelling in every record at once.
        cache["text"] = "\0" + "\0".join(texts)
        cache["starts"] = np.cumsum([1] + [len(t) + 1 for t in texts[:-1]])
        cache["mentions"] = {}
    return cache


def mentions(corpus, term):
    """Which records name the term, as a boolean array in the order of records()."""
    cache = records(corpus)
    if term not in cache["mentions"]:
        text = cache["text"]
        # A leading (?<!\w) would stop re from jumping to the literal start of a spelling and make the
        # scan 20 times slower, so the character before a match is tested here instead.
        at = [m.start() for a in spellings(term) for m in re.finditer(regex(a), text) if not re.match(r"\w", text[m.start() - 1])]
        hit = np.zeros(len(cache["ids"]), bool)
        hit[np.searchsorted(cache["starts"], at, side="right") - 1] = True
        cache["mentions"][term] = hit
    return cache["mentions"][term]


# The links of a hypothesis, each a pair of slots that one knowledge graph triple must name together. The structure is
# a pattern feature, tied to the biological model that has it and to the mechanism it brings.
CHAIN = [("biological_model", "mechanism"), ("mechanism", "ceramic"), ("ceramic", "process"), ("ceramic", "interlayer"),
         ("biological_model", "structure"), ("structure", "mechanism")]


def proposals(corpus, question, k=3):
    """Hypotheses whose every link a knowledge graph triple names, while no passage, caption or triple names all their terms.

    This is literature-based discovery [Swanson 1986]. A biology paper may tie nacre to tablets and tablets to crack
    deflection, and a materials paper crack deflection to silicon carbide, while no record names nacre, tablets, silicon
    carbide and graphite together. The combination novelty counts the same records, so a proposal scores above 0 there.
    A hypothesis holds one option per categorical field and one structure, a pattern feature. Every option and feature
    the question names must be in it. Only direct ink writing prints a structure (pattern_errors), so the process is
    direct ink writing, and no link from structure to process is asked for, since the pattern grammar decides what
    prints. The separation energy of the interlayer on the ceramic (md_interface) must be under a quarter of the
    cleavage energy of the ceramic (dft_cleavage), the He-Hutchinson ratio below which a crack deflects. Hypotheses
    rank by their weakest link, the number of triples that name it, and each differs from the ones before it in at
    least half of the free slots.
    """
    cache = records(corpus)
    options = {f: DESIGN[f].get("enum") or DESIGN[f]["examples"] for f in FIELDS} | {"structure": list(CARRIES)}
    terms = {o for v in options.values() for o in v}
    named = {o for o in terms if any(re.search(r"(?<!\w)" + regex(a), question.lower()) for a in spellings(o))}
    in_triples = {o: mentions(corpus, o) & cache["triple"] for o in terms}
    links, found = {}, []  # links holds the triples that name both terms of a pair
    for values in itertools.product(*options.values()):
        h = dict(zip(options, values))
        if (not named <= set(values) or h["process"] != "direct ink writing"
                or md_interface(h["ceramic"], h["interlayer"])["separation_energy_J_m2"]
                >= dft_cleavage(h["ceramic"])["cleavage_energy_J_m2"] / 4):
            continue
        for a, b in CHAIN:
            if (h[a], h[b]) not in links:
                links[h[a], h[b]] = np.flatnonzero(in_triples[h[a]] & in_triples[h[b]])
        n = [len(links[h[a], h[b]]) for a, b in CHAIN]
        # One record, a passage, a caption or a triple, that names every term has reported the hypothesis. A whole paper
        # is too coarse, since a review names most options of the field somewhere in its text.
        if min(n) and not np.logical_and.reduce([mentions(corpus, v) for v in values]).any():
            found.append((min(n), sum(n), h))
    free = [f for f in options if len({h[f] for _, _, h in found}) > 1]

    def example(a, b):
        """The triple shown for a link, preferably one whose head names one term and whose tail the other, an edge
        between the two, over one whose relation sentence names them in passing."""
        rows = [corpus["triples"][i - len(corpus["chunks"])] for i in links[a, b]]
        names = lambda text, term: any(re.search(r"(?<!\w)" + regex(x), text.lower()) for x in spellings(term))
        edge = lambda r: names(r["head"], a) and names(r["tail"], b) or names(r["head"], b) and names(r["tail"], a)
        return next((r for r in rows if edge(r)), rows[0])

    out, picked = [], []
    for weakest, _, h in sorted(found, key=lambda x: (-x[0], -x[1])):
        if all(2 * sum(h[f] != g[f] for f in free) >= len(free) for g in picked):
            picked.append(h)
            together = int(np.sum([mentions(corpus, v) for v in h.values()], axis=0).max())
            out.append({"design": {f: h[f] for f in FIELDS}, "structure": h["structure"], "weakest_link": weakest,
                        "most_terms_together": together, "links": [
                {"terms": [h[a], h[b]], "triples": len(links[h[a], h[b]]), "example": example(h[a], h[b])} for a, b in CHAIN]})
            if len(out) == k:
                break
    return out


def novelty(design, pattern, corpus, embed):
    """Combination novelty, semantic novelty and their mean for a design and its pattern. embed(texts) returns unit-norm vectors."""
    cache = records(corpus)
    text_only = ~corpus["is_figure"]
    if "reference" not in cache:
        # The best similarity to a text passage of the statements of 256 random hypotheses with random patterns.
        # A one-sentence statement lies far from every long passage, so a passage-to-passage reference would call
        # every hypothesis novel. Figure captions are left out, since the captions of one figure's panels are near copies.
        rng = np.random.default_rng(0)
        options = {k: DESIGN[k].get("enum") or DESIGN[k]["examples"] for k in FIELDS}
        designs = [{k: v[rng.integers(len(v))] for k, v in options.items()} for _ in range(256)]
        best = embed([statement(d, random_pattern(rng)) for d in designs]) @ corpus["vectors"][text_only].T
        cache["reference"] = best.max(1)

    terms = [design[k] for k in FIELDS if k in design] + features(pattern)
    counts = np.sum([mentions(corpus, t) for t in terms], axis=0)
    top = int(counts.max())
    with_top = np.flatnonzero((counts == top) & (counts > 0))
    combination = 1 - max(top - 1, 0) / (len(terms) - 1)

    s = statement(design, pattern)
    sims = np.where(text_only, corpus["vectors"] @ embed([s])[0], -np.inf)
    i = int(np.argmax(sims))
    semantic = float(np.mean(cache["reference"] > sims[i]))
    return {"statement": s, "score": (combination + semantic) / 2, "combination": combination, "semantic": semantic,
            "terms": terms, "max_terms_together": top, "records_with_max": len(with_top),
            "closest_records": [cache["ids"][j] for j in with_top[:3]], "nearest_passage": corpus["chunks"][i]["id"],
            "nearest_similarity": float(sims[i]),
            "method": f"Score is the mean of combination novelty, 1 - (m - 1)/(n - 1) with m the largest number of the n terms "
                      f"(design choices and pattern features, aliases included) that a single passage, figure caption or knowledge "
                      f"graph triple names, and semantic novelty, the fraction of {len(cache['reference'])} statements of random "
                      f"hypotheses with random patterns whose nearest text passage is more similar than that of the design "
                      f"statement (captions excluded)."}
