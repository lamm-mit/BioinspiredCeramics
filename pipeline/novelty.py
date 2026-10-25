"""How far the hypothesis of a design lies from the corpus, as a novelty score in [0, 1].

The hypothesis is the set of categorical choices of a design, such as nacre, alumina, graphite,
direct ink writing and crack deflection. Combination novelty asks how many of these choices one
record of the corpus already names together. A record is a passage, a figure caption or a knowledge
graph triple. Semantic novelty asks how close the nearest text passage comes to the hypothesis in one
sentence, ranked among the same sentences for random hypotheses of the design space, so that a
short sentence is compared with short sentences. The score is the mean of the two.
What depends only on the corpus is computed at the first call and kept in corpus["novelty"].
"""

import re

import numpy as np

from .design import DESIGN

# The categorical fields of a design, in the order the statement names them.
FIELDS = ["biological_model", "ceramic", "interlayer", "process", "mechanism"]
# The spellings a paper may use for each option. They match case-insensitively as whole words, with a
# space or a hyphen between words and an optional plural ending. A missed spelling makes a design look
# novel, and the search would then favor it, so the lists lean broad, as glass for silica glass.
ALIASES = {
    "nacre": ["nacre", "nacreous", "mother of pearl", "brick and mortar"],
    "conch shell": ["conch shell", "conch", "Strombus gigas", "crossed lamellar"],
    "bone": ["bone", "osteon", "osteonal", "Haversian"],
    "Bouligand": ["Bouligand", "helicoid", "helicoidal", "twisted plywood"],
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
}


def statement(design):
    """The hypothesis in one sentence, such as "A nacre-inspired alumina laminate with graphite interlayers,
    made by direct ink writing, toughened by crack deflection." """
    d = design
    s = (f"{d['biological_model']}-inspired " if "biological_model" in d else "") + f"{d.get('ceramic', 'ceramic')} laminate"
    s += f" with {d['interlayer']} interlayers" if "interlayer" in d else ""
    s += f", made by {d['process']}" if "process" in d else ""
    s += f", toughened by {d['mechanism']}" if "mechanism" in d else ""
    return ("An " if s[0] in "aeiouAEIOU" else "A ") + s + "."


def novelty(design, corpus, embed):
    """Combination novelty, semantic novelty and their mean for a design. embed(texts) returns unit-norm vectors."""
    cache = corpus.setdefault("novelty", {})
    text_only = ~corpus["is_figure"]
    if not cache:
        texts = [c["text"].lower() for c in corpus["chunks"]]
        texts += [f"{t['head']} {t['relation']} {t['tail']}".lower() for t in corpus["triples"]]
        cache["ids"] = [c["id"] for c in corpus["chunks"]] + [f"kg:{i}" for i in range(len(corpus["triples"]))]
        # All records in one string, so that one regex scan finds a spelling in every record at once.
        cache["text"] = "\0" + "\0".join(texts)
        cache["starts"] = np.cumsum([1] + [len(t) + 1 for t in texts[:-1]])
        cache["mentions"] = {}
        # The best similarity to a text passage of the statements of 256 random hypotheses. A one-sentence
        # statement lies far from every long passage, so a passage-to-passage reference would call every
        # hypothesis novel. Figure captions are left out, since the captions of one figure's panels are near copies.
        rng = np.random.default_rng(0)
        options = {k: DESIGN[k]["enum"] for k in FIELDS}
        designs = [{k: v[rng.integers(len(v))] for k, v in options.items()} for _ in range(256)]
        best = embed([statement(d) for d in designs]) @ corpus["vectors"][text_only].T
        cache["reference"] = best.max(1)

    terms = [design[k] for k in FIELDS if k in design]
    text = cache["text"]
    for term in set(terms) - cache["mentions"].keys():
        # A leading (?<!\w) would stop re from jumping to the literal start of a spelling and make the
        # scan 20 times slower, so the character before a match is tested here instead.
        spellings = [re.escape(a.lower()).replace(r"\ ", r"[\s-]+").replace(r"\-", r"[\s-]+") for a in ALIASES.get(term, [term])]
        at = [m.start() for p in spellings for m in re.finditer(p + r"(?:e?s)?(?!\w)", text) if not re.match(r"\w", text[m.start() - 1])]
        hit = np.zeros(len(cache["ids"]), bool)
        hit[np.searchsorted(cache["starts"], at, side="right") - 1] = True
        cache["mentions"][term] = hit
    counts = np.sum([cache["mentions"][t] for t in terms], axis=0)
    top = int(counts.max())
    with_top = np.flatnonzero((counts == top) & (counts > 0))
    combination = 1 - max(top - 1, 0) / (len(terms) - 1)

    s = statement(design)
    sims = np.where(text_only, corpus["vectors"] @ embed([s])[0], -np.inf)
    i = int(np.argmax(sims))
    semantic = float(np.mean(cache["reference"] > sims[i]))
    return {"statement": s, "score": (combination + semantic) / 2, "combination": combination, "semantic": semantic,
            "terms": terms, "max_terms_together": top, "records_with_max": len(with_top),
            "closest_records": [cache["ids"][j] for j in with_top[:3]], "nearest_passage": corpus["chunks"][i]["id"],
            "nearest_similarity": float(sims[i]),
            "method": f"Score is the mean of combination novelty, 1 - (m - 1)/(n - 1) with m the largest number of the n design "
                      f"terms (aliases included) that a single passage, figure caption or knowledge graph triple names, and semantic "
                      f"novelty, the fraction of {len(cache['reference'])} statements of random hypotheses whose nearest text passage "
                      f"is more similar than that of the design statement (captions excluded)."}
