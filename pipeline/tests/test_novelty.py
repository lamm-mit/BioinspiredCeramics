# Tests for the novelty score, on a small invented corpus with a fake embedding. python -m pipeline.tests.test_novelty
import json, math, re, time, zlib
from pathlib import Path
import numpy as np
from pipeline.novelty import ALIASES, features, novelty, proposals, statement
from pipeline.pattern import describe

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
FLAT = {}
NACRE = {"break_length_h": 20, "interlayer_placement": "shell"}
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
         statement(SAME, FLAT),
         "Robocast alumina platelets, coextruded in a brick-and-mortar arrangement, deflect cracks."]
chunks = [{"id": f"p{i}", "source": "invented", "text": t} for i, t in enumerate(TEXTS)]
chunks.append({"id": "f0", "source": "invented", "text": statement(CAPTION, FLAT), "image": "f0.png"})
triples = [{"head": "silicon carbide", "relation": "is printed by", "tail": "binder jetting", "source": "invented"},
           {"head": "Bouligand structure", "relation": "toughens by", "tail": "crack twisting", "source": "invented"}]
CALLS = []  # the texts of every embed call
corpus = {"folder": Path("."), "chunks": chunks, "vectors": embed([c["text"] for c in chunks]), "embed_model": "fake",
          "is_figure": np.array(["image" in c for c in chunks]), "triples": triples}
def recorded(texts):
    CALLS.append(texts)
    return embed(texts)

print(sum(map(len, ALIASES.values())), "spellings for", len(ALIASES), "options and features")
check("statement reads as one sentence", statement(FULL, FLAT) == "An alumina laminate inspired by nacre, with graphite interlayers, made by "
      "direct ink writing, toughened by crack deflection, printed with straight roads, interlayer sheets between plies.", statement(FULL, FLAT))
check("statement ends with the pattern in words", statement(FULL, NACRE).endswith(f"printed with {describe(NACRE)}.")
      and "tablets 20 layer thicknesses long" in statement(FULL, NACRE), statement(FULL, NACRE))
r = {name: novelty(d, FLAT, corpus, recorded) for name, d in [("full", FULL), ("apart", APART), ("same", SAME), ("caption", CAPTION)]}
check("full combination in one passage gives combination 0", r["full"]["combination"] == 0 and r["full"]["max_terms_together"] == 5 and r["full"]["closest_records"] == ["p0"], r["full"])
check("aliases count, as robocasting for direct ink writing and Al2O3 for alumina", "direct ink writing" not in TEXTS[0] and "alumina" not in TEXTS[0].lower(), TEXTS[0])
check("terms that never share a record give combination 1", r["apart"]["combination"] == 1 and r["apart"]["max_terms_together"] == 1, r["apart"])
sic = novelty({"ceramic": "silicon carbide", "process": "binder jetting"}, FLAT, corpus, embed)
check("a triple counts as a record", sic["combination"] == 0 and sic["closest_records"] == ["kg:0"], sic)
basic = novelty({"ceramic": "silicon carbide", "interlayer": "epoxy"}, FLAT, corpus, embed)
check("SiC matches whole words only, so basic and classic do not count", basic["max_terms_together"] == 1 and basic["records_with_max"] == 5, basic)
bridge = novelty({"interlayer": "PMMA", "mechanism": "crack bridging"}, FLAT, corpus, embed)
check("a hyphen matches a space, and case is ignored", bridge["combination"] == 0 and "p9" in bridge["closest_records"], bridge)
novelty({"biological_model": "conch shell", "mechanism": "crack deflection"}, FLAT, corpus, embed)
conch = np.flatnonzero(corpus["novelty"]["mentions"]["conch shell"]).tolist()
check("a plural counts, as Conch shells for conch shell", conch == [6], conch)
check("score is the mean of the two parts, all in [0, 1]", all(math.isclose(x["score"], (x["combination"] + x["semantic"]) / 2)
      and all(0 <= x[k] <= 1 for k in ["score", "combination", "semantic"]) for x in r.values()), [(x["combination"], x["semantic"]) for x in r.values()])
check("a statement identical to a passage has semantic novelty near 0", r["same"]["semantic"] < 0.1 and r["same"]["nearest_passage"] == "p10"
      and r["same"]["nearest_similarity"] > 0.999, r["same"])
check("a caption counts for the combination but not for the nearest passage", r["caption"]["combination"] == 0 and "f0" in r["caption"]["closest_records"]
      and r["caption"]["nearest_passage"] != "f0", r["caption"])
check("the result is JSON", json.loads(json.dumps(r)) == r)

# Pattern features
PATTERNS = {"flat": ({}, []), "nacre": (NACRE, ["tablets", "core-shell roads"]),
            "Bouligand": ({"ply_rotation_deg": 15, "interlayer_placement": "shell"}, ["helicoidal plies", "core-shell roads"]),
            "crossed": ({"road_angle_deg": -30, "ply_rotation_deg": 60, "ply_rotation_alternates": True, "interlayer_placement": "shell"},
                        ["crossed lamellae", "core-shell roads"]),
            "rings": ({"road_path": "rings", "ring_spacing_w": 8, "interlayer_placement": "shell"}, ["concentric rings", "core-shell roads"]),
            "waves": ({"road_path": "waves", "wave_amplitude_w": 1, "wave_length_w": 10, "interlayer_placement": "shell"},
                      ["wavy interfaces", "core-shell roads"]),
            "wavy plies": ({"ply_wave_amplitude_h": 0.5, "ply_wave_length_h": 30, "break_length_h": 20}, ["tablets", "wavy interfaces"]),
            "graded": ({"modulus_profile": "linear", "outer_layer_modulus_GPa": 250}, ["graded modulus"]),
            "angled roads": ({"road_angle_deg": 45}, [])}
found = {name: features(p) for name, (p, _) in PATTERNS.items()}
check("a pattern adds a term for each feature it uses and none for the rest", all(found[n] == want for n, (_, want) in PATTERNS.items()), found)
check("every feature has spellings", all(t in ALIASES for _, want in PATTERNS.values() for t in want))
tab = novelty(FULL, NACRE, corpus, embed)
check("pattern terms join the design terms, and their aliases count, as platelets for tablets and coextruded for core-shell",
      tab["terms"] == list(FULL.values()) + ["tablets", "core-shell roads"] and tab["max_terms_together"] == 6 and tab["closest_records"] == ["p11"]
      and math.isclose(tab["combination"], 1 - 5 / 6), tab)
check("p11 holds four design terms, so the flat design still has p0 alone at the top", r["full"]["closest_records"] == ["p0"])

# A biological model in free text
free = novelty(FULL | {"biological_model": "red abalone nacre"}, FLAT, corpus, embed)
mentions = corpus["novelty"]["mentions"]
check("a free-text model that names a listed model counts that model's records", (mentions["red abalone nacre"] == mentions["nacre"]).all()
      and free["combination"] == r["full"]["combination"] and "inspired by red abalone nacre," in free["statement"], np.flatnonzero(mentions["red abalone nacre"]))
odd = novelty(FULL | {"biological_model": "C. elegans (cuticle) [?]"}, FLAT, corpus, embed)
check("any other string matches as written and leaves the score in [0, 1]", not mentions["C. elegans (cuticle) [?]"].any()
      and 0 <= odd["score"] <= 1 and odd["terms"][0] == "C. elegans (cuticle) [?]", odd["terms"])

# The reference of random hypotheses
reference = CALLS[0]
check("the reference statements are 256 random hypotheses in the statement grammar, with random patterns",
      len(reference) == 256 and all(re.fullmatch(r"An? .+ laminate inspired by .+, printed with .+\.", s) for s in reference)
      and len(set(s.split("printed with ")[1] for s in reference)) > 200
      and all(any(word in s for s in reference) for word in ["tablets", "Bouligand stack", "crossed", "concentric rings", "wavy", "graded", "shells"]),
      reference[:2])
nearest = corpus["novelty"]["reference"]
calls = []
again = novelty(APART, FLAT, corpus, lambda t: calls.append(len(t)) or embed(t))
check("the reference of random hypotheses is cached, and a call embeds only the statement", corpus["novelty"]["reference"] is nearest
      and calls == [1] and len(nearest) == 256 and again == r["apart"], (calls, len(nearest)))
corpus["novelty"]["reference"] = np.array([2.0])
check("the cached reference is the one used", novelty(FULL, FLAT, corpus, embed)["semantic"] == 1, corpus["novelty"]["reference"])

# Knowledge graph proposals, on corpora of a few triples. Paper ids in a triple use / and in a passage id _.
def kg(triples, passages=()):
    return {"chunks": [{"id": f"{paper}#0", "source": paper, "text": text} for paper, text in passages],
            "triples": [{"head": h, "relation": r, "tail": tail, "source": paper} for paper, h, r, tail in triples]}
LINKS = [("paper/a", "nacre", "toughens by", "crack deflection"), ("paper/b", "crack deflection", "is seen in", "silicon carbide laminates"),
         ("paper/c", "silicon carbide", "is printed by", "direct ink writing"), ("paper/d", "silicon carbide", "is joined by", "graphite interlayers"),
         ("paper/g", "nacre", "is built of", "aragonite tablets"), ("paper/h", "tablets", "deflect", "cracks")]
OPEN = "Propose a bioinspired layered ceramic made by additive manufacturing."
NSG = {"biological_model": "nacre", "ceramic": "silicon carbide", "interlayer": "graphite", "process": "direct ink writing", "mechanism": "crack deflection"}
p = proposals(kg(LINKS), OPEN)
check("a hypothesis whose six links are six separate triples is proposed, with its structure and the triple of each link",
      [(x["design"], x["structure"]) for x in p] == [(NSG, "tablets")] and [l["triples"] for l in p[0]["links"]] == [1] * 6
      and [l["example"]["source"] for l in p[0]["links"]] == ["paper/a", "paper/b", "paper/c", "paper/d", "paper/g", "paper/h"], p)
check("a passage that names every term reports the hypothesis",
      proposals(kg(LINKS, [("paper_x", "Nacre-like silicon carbide tablets with graphite, by direct ink writing, deflect cracks.")]), OPEN) == [], "")
split = proposals(kg(LINKS, [("paper_a", "Silicon carbide tablets with graphite, by direct ink writing.")]), OPEN)
check("terms split between two records of one paper do not, and the most terms one record names is given",
      [x["design"] for x in split] == [NSG] and split[0]["most_terms_together"] == 4 and p[0]["most_terms_together"] == 2, split)
edge = proposals(kg([("paper/z", "SiC laminates", "are toughened by", "crack deflection along nacre-like interfaces")] + LINKS), OPEN)
check("a triple whose head and tail name the two terms is shown over one whose relation names them",
      edge[0]["links"][0]["example"]["source"] == "paper/a" and edge[0]["links"][0]["triples"] == 2, edge[0]["links"][0])
check("an option the question names must be in the hypothesis", [x["design"] for x in proposals(kg(LINKS), OPEN + " Use nacre.")] == [NSG]
      and proposals(kg(LINKS), "Design a layered alumina.") == [], "")
check("only direct ink writing prints a structure, so a process link to binder jetting alone gives no proposal",
      proposals(kg([x if x[0] != "paper/c" else ("paper/c", "silicon carbide", "is printed by", "binder jetting") for x in LINKS]), OPEN) == [], "")
weak = proposals(kg(LINKS + [("paper/e", "silicon carbide", "is joined by", "silica glass interlayers")]), OPEN)
check("an interlayer needs under a quarter of the cleavage energy of the ceramic, so silica glass is no interlayer for SiC",
      [x["design"] for x in weak] == [NSG], [x["design"] for x in weak])
# Bouligand plies name the model Bouligand, the structure helicoidal plies and crack deflection at once.
BOULIGAND = LINKS + [("paper/f", "Bouligand plies", "show", "crack deflection")] * 2
two = proposals(kg(BOULIGAND), OPEN)
check("hypotheses rank by their weakest link, then by all their links", [(x["design"]["biological_model"], x["structure"]) for x in two]
      == [("Bouligand", "helicoidal plies"), ("nacre", "tablets")] and [x["weakest_link"] for x in two] == [1, 1], [x["design"] for x in two])
check("a structure the question names is the structure of every proposal",
      [x["structure"] for x in proposals(kg(BOULIGAND), OPEN + " Use helicoidal plies.")] == ["helicoidal plies"], "")

# Timing on the eight-paper corpus, read in place, with random unit vectors as passage vectors.
folder = Path("/Users/leemarom/MIT Dropbox/Lee Marom/DSO/M2/examples/corpus")
if folder.exists():
    chunks = [json.loads(line) for line in open(folder / "chunks.jsonl")]
    v = np.random.default_rng(1).normal(size=(len(chunks), D)).astype(np.float32)
    real = {"folder": folder, "chunks": chunks, "vectors": v / np.linalg.norm(v, axis=1, keepdims=True), "embed_model": "fake",
            "is_figure": np.array(["image" in c for c in chunks]), "triples": [json.loads(line) for line in open(folder / "kg.jsonl")]}
    t0 = time.perf_counter()
    first = novelty(FULL, NACRE, real, embed)
    t1 = time.perf_counter()
    second = novelty(SAME, PATTERNS["Bouligand"][0], real, embed)
    t2 = time.perf_counter()
    check("eight-paper corpus", 0 <= first["score"] <= 1 and 0 <= second["score"] <= 1,
          f"{len(chunks)} chunks, {len(real['triples'])} triples, first call {t1 - t0:.3f} s, next call with new terms {t2 - t1:.3f} s")
    for x in (first, second):
        print("    ", {k: x[k] for k in ["terms", "combination", "max_terms_together", "records_with_max", "closest_records"]})
print("FAILED" if bad else "ALL PASS", bad)
