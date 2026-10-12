"""Read a corpus of passages and figures, and a knowledge graph of triples.

A corpus folder holds chunks.jsonl with one record per line, such as
  {"id": "10.1038_nmat4089#12", "source": "10.1038/nmat4089", "text": "..."}
A figure is a chunk whose text is the caption and whose "image" is a path relative to the folder.
An optional kg.jsonl holds one triple per line, such as
  {"head": "nacre", "relation": "Nacre deflects cracks along ...", "tail": "crack deflection", "source": "..."}
"""

import itertools
import json
import re
from pathlib import Path

import numpy as np

from .llm import embed


def load(folder, embed_model):
    folder = Path(folder)
    chunks = [json.loads(line) for line in open(folder / "chunks.jsonl")]
    cache = folder / f"vectors-{embed_model.replace(':', '-').replace('/', '-')}.npy"
    vectors = np.load(cache) if cache.exists() else []
    if len(vectors) != len(chunks):  # no cache yet, or the corpus was rebuilt
        # A single long paragraph, such as a table, can exceed the 2,000-character chunk size. Embedding
        # only its start keeps every vector on a similar amount of text.
        vectors = embed(embed_model, [c["text"][:2000] for c in chunks])
        np.save(cache, vectors)
    kg = folder / "kg.jsonl"
    return {"folder": folder, "chunks": chunks, "vectors": vectors, "embed_model": embed_model,
            "is_figure": np.array(["image" in c for c in chunks]),
            "triples": [json.loads(line) for line in open(kg)] if kg.exists() else []}


def search(corpus, query, k, figures=False, skip=(), pattern=None):
    """Top-k passages, or top-k figures by caption, by cosine similarity to the query.

    Ids in skip are left out. With a pattern, only chunks whose text matches it are returned.
    """
    scores = corpus["vectors"] @ embed(corpus["embed_model"], [query])[0]
    scores[corpus["is_figure"] != figures] = -np.inf
    best = (corpus["chunks"][i] for i in np.argsort(-scores) if np.isfinite(scores[i]))
    keep = (c for c in best if c["id"] not in skip and (pattern is None or re.search(pattern, c["text"])))
    return list(itertools.islice(keep, k))


def neighbors(corpus, entities, k=3):
    """For each entity, the k triples that mention it together with the most other entities."""
    terms = [e.lower() for e in entities]
    texts = [f"{t['head']} {t['relation']} {t['tail']}".lower() for t in corpus["triples"]]
    ranked = sorted(range(len(texts)), key=lambda i: -sum(x in texts[i] for x in terms))
    picked = {i: corpus["triples"][i] for term in terms for i in [j for j in ranked if term in texts[j]][:k]}
    return list(picked.values())
