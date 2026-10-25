# Tests for the novelty score, on a small invented corpus with a fake embedding. python -m pipeline.tests.test_novelty
import json, math, re, time, zlib
from pathlib import Path
import numpy as np
from pipeline.novelty import ALIASES, novelty, statement

bad = 0
def check(name, ok, detail=""):
    global bad
    bad += not ok
    print("ok  " if ok else "FAIL", name, "|", str(detail)[:150])

D = 256
def embed(texts):
    """Hashed bag of words, unit norm. Deterministic, since crc32 is not salted per process as hash() is."""
    v = np.zeros((len(texts), D), np.float32)
    for row, t in zip(v, texts):
        for w in re.findall(r"\w+", t.lower()):
            row[zlib.crc32(w.encode()) % D] += 1
    return v / np.linalg.norm(v, axis=1, keepdims=True)

FULL = {"biological_model": "nacre", "ceramic": "alumina", "interlayer": "graphite", "process": "direct ink writing", "mechanism": "crack deflection"}
APART = {"biological_model": "Bouligand", "ceramic": "silica glass", "interlayer": "epoxy", "process": "binder jetting", "mechanism": "compressive residual stress"}
SAME = {"biological_model": "bone", "ceramic": "silicon carbide", "interlayer": "PMMA", "process": "vat photopolymerization", "mechanism": "crack bridging"}
CAPTION = {"biological_model": "bone", "ceramic": "alumina", "interlayer": "PMMA", "process": "direct ink writing", "mechanism": "crack bridging"}
TEXTS = ["Robocasting of Al2O3 pastes with graphite sheets gives nacre-like laminates, and cracks deflect along the sheets.",
         "Stereolithography prints silicon carbide green bodies with 50 um resolution.",
         "Helicoidal fibers in the mantis shrimp club resist impact.",
         "Epoxy infiltrated into a porous scaffold acts as a compliant phase.",
         "Binder jetting deposits a liquid binder onto a powder bed one layer at a time.",
         "Compressive residual stresses in the outer layers arrest surface cracks.",
         "Cortical tissue holds osteons whose cement lines stop microcracks, as crossed-lamellar Conch shells do.",
         "Poly(methyl methacrylate) is the soft phase between glass tablets.",
         "A basic epoxy primer coats the classic steel substrate.",
         "Crack-bridging ligaments of PMMA span the opening behind the tip.",
         statement(SAME)]
chunks = [{"id": f"p{i}", "source": "invented", "text": t} for i, t in enumerate(TEXTS)]
chunks.append({"id": "f0", "source": "invented", "text": statement(CAPTION), "image": "f0.png"})
triples = [{"head": "silicon carbide", "relation": "is printed by", "tail": "binder jetting", "source": "invented"},
           {"head": "Bouligand structure", "relation": "toughens by", "tail": "crack twisting", "source": "invented"}]
corpus = {"folder": Path("."), "chunks": chunks, "vectors": embed([c["text"] for c in chunks]), "embed_model": "fake",
          "is_figure": np.array(["image" in c for c in chunks]), "triples": triples}

print(sum(map(len, ALIASES.values())), "spellings for", len(ALIASES), "options")
check("statement reads as one sentence", statement(FULL) == "A nacre-inspired alumina laminate with graphite interlayers, made by direct ink writing, toughened by crack deflection.", statement(FULL))
r = {name: novelty(d, corpus, embed) for name, d in [("full", FULL), ("apart", APART), ("same", SAME), ("caption", CAPTION)]}
check("full combination in one passage gives combination 0", r["full"]["combination"] == 0 and r["full"]["max_terms_together"] == 5 and r["full"]["closest_records"] == ["p0"], r["full"])
check("aliases count, as robocasting for direct ink writing and Al2O3 for alumina", "direct ink writing" not in TEXTS[0] and "alumina" not in TEXTS[0].lower(), TEXTS[0])
check("terms that never share a record give combination 1", r["apart"]["combination"] == 1 and r["apart"]["max_terms_together"] == 1, r["apart"])
sic = novelty({"ceramic": "silicon carbide", "process": "binder jetting"}, corpus, embed)
check("a triple counts as a record", sic["combination"] == 0 and sic["closest_records"] == ["kg:0"], sic)
basic = novelty({"ceramic": "silicon carbide", "interlayer": "epoxy"}, corpus, embed)
check("SiC matches whole words only, so basic and classic do not count", basic["max_terms_together"] == 1 and basic["records_with_max"] == 5, basic)
bridge = novelty({"interlayer": "PMMA", "mechanism": "crack bridging"}, corpus, embed)
check("a hyphen matches a space, and case is ignored", bridge["combination"] == 0 and "p9" in bridge["closest_records"], bridge)
novelty({"biological_model": "conch shell", "mechanism": "crack deflection"}, corpus, embed)
conch = np.flatnonzero(corpus["novelty"]["mentions"]["conch shell"]).tolist()
check("a plural counts, as Conch shells for conch shell", conch == [6], conch)
check("score is the mean of the two parts, all in [0, 1]", all(math.isclose(x["score"], (x["combination"] + x["semantic"]) / 2)
      and all(0 <= x[k] <= 1 for k in ["score", "combination", "semantic"]) for x in r.values()), [(x["combination"], x["semantic"]) for x in r.values()])
check("a statement identical to a passage has semantic novelty near 0", r["same"]["semantic"] < 0.1 and r["same"]["nearest_passage"] == "p10"
      and r["same"]["nearest_similarity"] > 0.999, r["same"])
check("a caption counts for the combination but not for the nearest passage", r["caption"]["combination"] == 0 and "f0" in r["caption"]["closest_records"]
      and r["caption"]["nearest_passage"] != "f0", r["caption"])
check("the result is JSON", json.loads(json.dumps(r)) == r)

nearest = corpus["novelty"]["reference"]
calls = []
again = novelty(APART, corpus, lambda t: calls.append(len(t)) or embed(t))
check("the reference of random hypotheses is cached, and a call embeds only the statement", corpus["novelty"]["reference"] is nearest
      and calls == [1] and len(nearest) == 256 and again == r["apart"], (calls, len(nearest)))
corpus["novelty"]["reference"] = np.array([2.0])
check("the cached reference is the one used", novelty(FULL, corpus, embed)["semantic"] == 1, corpus["novelty"]["reference"])

# Timing on the eight-paper corpus, read in place, with random unit vectors as passage vectors.
folder = Path("/Users/leemarom/MIT Dropbox/Lee Marom/DSO/M2/examples/corpus")
if folder.exists():
    chunks = [json.loads(line) for line in open(folder / "chunks.jsonl")]
    v = np.random.default_rng(1).normal(size=(len(chunks), D)).astype(np.float32)
    real = {"folder": folder, "chunks": chunks, "vectors": v / np.linalg.norm(v, axis=1, keepdims=True), "embed_model": "fake",
            "is_figure": np.array(["image" in c for c in chunks]), "triples": [json.loads(line) for line in open(folder / "kg.jsonl")]}
    t0 = time.perf_counter()
    first = novelty(FULL, real, embed)
    t1 = time.perf_counter()
    second = novelty(SAME, real, embed)
    t2 = time.perf_counter()
    check("eight-paper corpus", 0 <= first["score"] <= 1 and 0 <= second["score"] <= 1,
          f"{len(chunks)} chunks, {len(real['triples'])} triples, first call {t1 - t0:.3f} s, next call with new terms {t2 - t1:.3f} s")
    for x in (first, second):
        print("    ", {k: x[k] for k in ["combination", "max_terms_together", "records_with_max", "closest_records"]})
print("FAILED" if bad else "ALL PASS", bad)
