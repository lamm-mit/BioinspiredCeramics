"""
06_dedup_kg.py — Embedding-based node deduplication and graph assembly.

The triplets emitted by `05_extract_kg.py` contain many synonymous node
names (e.g. "alumina", "Al2O3", "aluminum oxide" all refer to the same
concept). This script merges them into a single canonical node by:

  1. Embedding every unique node name with Google's EmbeddingGemma-300m
     using the "STS" (Semantic Textual Similarity) prompt — the prompt
     designed for entity-similarity tasks. Avoid the "Clustering" prompt:
     it groups by topic, not entity identity, and produces nonsense
     merges like ("3d printing", "guidance", "questions").

  2. Finding clusters of near-duplicates with a chunked GPU cosine search
     via `sentence_transformers.util.community_detection` — the same
     algorithm used by the sentence-transformers maintainers for entity
     dedup at scale.

  3. Picking a canonical name per cluster (highest in-graph degree wins,
     shortest label as tiebreaker), then rebuilding the merged graph
     with `networkx`.

  4. Detecting thematic communities on the deduplicated graph using
     Louvain modularity (`python-louvain`).

Default threshold is 0.88 with the STS prompt — empirically the best
balance for EmbeddingGemma on a 400K-node corpus: synonyms still merge
(alumina ↔ aluminum oxide ≈ 0.92) while specific concepts stay separate
(nanoparticles vs nanostructures, hydroxyapatite vs collagen). Compare
to `lamm-mit/GraphAgents` which uses 0.9 with nomic-embed; Gemma's STS
distribution shifts synonym scores down by ~0.02 vs nomic.

Usage:
    python knowledge_graph/06_dedup_kg.py \\
        --triplet-dirs data/kg/triplets \\
        --output-dir data/kg/dedup

    # Merge KGs from two corpora into one deduplicated graph
    python knowledge_graph/06_dedup_kg.py \\
        --triplet-dirs data/kg/bioinspired/triplets \\
                       data/kg/ceramic_am/triplets \\
        --output-dir data/kg/dedup

    # Iterate on threshold without re-embedding
    python knowledge_graph/06_dedup_kg.py \\
        --triplet-dirs data/kg/triplets \\
        --output-dir data/kg/dedup \\
        --threshold 0.88 \\
        --reuse-embeddings

Outputs to `--output-dir`:
  * node_embeddings.npy            cached Gemma embeddings (~1.3 GB / 400K nodes)
  * nodes.txt                      node-name list, one per line
  * merge_clusters.tsv             every merged cluster with its members
  * canonical_map.tsv              original_name -> canonical_name mapping
  * knowledge_graph_dedup.graphml  deduplicated graph with Louvain communities

Requirements:
  * sentence-transformers >= 3.0
  * EmbeddingGemma is a gated HF model — run `huggingface-cli login` once
  * networkx, python-louvain
  * CUDA GPU strongly recommended

Credits: Algorithm follows the entity-dedup pattern used in
`lamm-mit/GraphAgents` (LAMM lab), with `community_detection` from
the sentence-transformers library standing in for the brute-force N×N
similarity matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import community_detection


_NUMERIC_CHARS = set("0123456789.,+-eE")
_UNIT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ%/μ°")


def _is_pure_quantity(name: str) -> bool:
    """Return True if `name` is a pure numeric quantity like '2 mm', '5%', '10.5 gpa'.

    Implemented without regex to avoid catastrophic backtracking on
    pathological inputs.
    """
    tokens = name.strip().split()
    if not tokens:
        return False
    # First token must contain at least one digit and only numeric chars
    if not any(c.isdigit() for c in tokens[0]):
        return False
    if not all(c in _NUMERIC_CHARS for c in tokens[0]):
        return False
    # Remaining tokens must be short unit-like words (no digits)
    for t in tokens[1:]:
        if len(t) > 8:
            return False
        if not all(c in _UNIT_CHARS for c in t):
            return False
    return True


def is_pathological(name: str) -> bool:
    """Return True if a node name should be excluded from embedding/merging."""
    if not name or len(name) < 2 or len(name) > 200:
        return True
    # LaTeX / math notation
    if "$" in name or "\\" in name:
        return True
    # Pure quantity expressions ("2 mm", "5%", "10 gpa")
    if _is_pure_quantity(name):
        return True
    # Must have at least 2 alphabetic characters somewhere
    alpha = sum(c.isalpha() for c in name)
    if alpha < 2:
        return True
    return False


def clean_nodes(nodes: list[str]) -> tuple[list[str], np.ndarray]:
    """Filter pathological nodes. Returns (kept_nodes, mask_into_original)."""
    mask = np.array([not is_pathological(n) for n in nodes], dtype=bool)
    kept = [n for n, k in zip(nodes, mask) if k]
    return kept, mask


def load_all_triplets(dirs: list[Path]) -> tuple[list[dict], list[str]]:
    """Read every per-paper triplet JSON from the given directories."""
    triplets: list[dict] = []
    for d in dirs:
        files = list(d.glob("*.json"))
        print(f"  {d}: {len(files)} files")
        for f in files:
            try:
                triplets.extend(json.loads(f.read_text()))
            except Exception:
                continue

    # Collect all unique node names
    nodes = set()
    for t in triplets:
        n1 = str(t.get("node_1", "")).lower().strip()
        n2 = str(t.get("node_2", "")).lower().strip()
        if n1:
            nodes.add(n1)
        if n2:
            nodes.add(n2)

    # Drop pathological / very long / single-char
    nodes = sorted(n for n in nodes if 1 < len(n) <= 200)
    return triplets, nodes


def embed_nodes(
    model: SentenceTransformer,
    nodes: list[str],
    batch_size: int,
    prompt_name: str | None = "STS",
) -> np.ndarray:
    """Encode node names. Returns L2-normalized array.

    Default prompt is "STS" (Semantic Textual Similarity) — appropriate for
    entity deduplication where we want "alumina" ≈ "Al2O3" but
    "alumina" ≉ "3D printing". The "Clustering" prompt produces broad
    thematic groupings unsuitable for synonym dedup. Set prompt_name=None
    to skip the prompt entirely.
    """
    print(f"Embedding {len(nodes):,} nodes (batch_size={batch_size})...", flush=True)
    t0 = time.time()
    encode_kwargs = dict(
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    if prompt_name and hasattr(model, "prompts") and prompt_name in (model.prompts or {}):
        encode_kwargs["prompt_name"] = prompt_name
    embeddings = model.encode(nodes, **encode_kwargs)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)
    return embeddings


def cluster_and_canonicalize(
    embeddings: np.ndarray,
    nodes: list[str],
    degrees: Counter,
    threshold: float,
    batch_size: int,
    output_dir: Path,
) -> tuple[dict[str, str], list[list[str]]]:
    """Cluster nodes by embedding similarity using sentence-transformers'
    `community_detection`, then pick a canonical name per cluster.

    Algorithm comes from huggingface/sentence-transformers
    util/retrieval.py: chunked GPU cosine -> per-row threshold mask -> topk
    extraction -> greedy overlap removal. Battle-tested for entity dedup at
    scale.

    Canonical name = highest graph-degree member, then shortest, then alpha.
    """
    print(
        f"Running community_detection (threshold={threshold}, "
        f"batch_size={batch_size})...",
        flush=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_t = torch.from_numpy(embeddings).to(device)
    print(f"  device={device}, embeddings: {tuple(emb_t.shape)}", flush=True)

    t0 = time.time()
    clusters_idx = community_detection(
        emb_t,
        threshold=threshold,
        min_community_size=2,  # any pair of similar nodes should merge
        batch_size=batch_size,
        show_progress_bar=True,
    )
    print(
        f"  found {len(clusters_idx):,} clusters in {time.time() - t0:.1f}s",
        flush=True,
    )

    # Free GPU memory before downstream CPU-heavy work
    del emb_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clusters_named = [[nodes[i] for i in c] for c in clusters_idx]

    canonical: dict[str, str] = {}
    for members in clusters_named:
        # Highest degree wins; tiebreak by shorter label, then alphabetical
        canon = min(members, key=lambda m: (-degrees.get(m, 0), len(m), m))
        for m in members:
            if m != canon:
                canonical[m] = canon

    # Write inspectable cluster report (replaces per-pair merge_pairs.tsv)
    report_path = output_dir / "merge_clusters.tsv"
    with open(report_path, "w") as f:
        f.write("cluster_id\tsize\tcanonical\tmembers\n")
        for cid, members in enumerate(clusters_named):
            canon = canonical.get(members[0], members[0])
            f.write(f"{cid}\t{len(members)}\t{canon}\t{' | '.join(members)}\n")
    print(f"  cluster report -> {report_path}", flush=True)

    return canonical, clusters_named


def build_graph(triplets: list[dict], canonical: dict[str, str]) -> nx.DiGraph:
    """Assemble final deduplicated NetworkX graph."""
    G = nx.DiGraph()
    for t in triplets:
        n1 = str(t.get("node_1", "")).lower().strip()
        n2 = str(t.get("node_2", "")).lower().strip()
        edge = str(t.get("edge", "")).strip()
        if not n1 or not n2:
            continue
        n1 = canonical.get(n1, n1)
        n2 = canonical.get(n2, n2)
        if n1 == n2:
            continue
        G.add_edge(n1, n2, title=edge, paper=t.get("paper", ""))
    return G


def run_louvain(G: nx.DiGraph) -> dict[str, int]:
    """Run Louvain community detection on the undirected version of G."""
    try:
        import community as community_louvain
    except ImportError:
        print("Warning: python-louvain not installed, skipping community detection")
        return {}
    UG = G.to_undirected()
    print(f"Running Louvain on {UG.number_of_nodes()} nodes...")
    partition = community_louvain.best_partition(UG, random_state=42)
    n_communities = len(set(partition.values()))
    print(f"  {n_communities} communities found")
    return partition


def main() -> None:
    parser = argparse.ArgumentParser(description="Dedup KG nodes via Gemma embeddings.")
    parser.add_argument(
        "--triplet-dirs", type=Path, nargs="+", required=True,
        help="One or more directories containing per-paper triplet JSON files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write deduplicated graph and reports.",
    )
    parser.add_argument(
        "--model", default="google/embeddinggemma-300m",
        help="HuggingFace model name.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.88,
        help="Cosine similarity threshold for merging (default 0.88, "
        "tuned for STS-prompt EmbeddingGemma — synonyms typically score "
        "0.88-0.95, unrelated entities <0.7). Lower to 0.85 to catch more "
        "typographically-distant variants at the cost of some over-merging.",
    )
    parser.add_argument(
        "--prompt", default="STS",
        help="EmbeddingGemma prompt name: STS (default, for synonym dedup), "
        "Retrieval-document (cleaner separation, slightly lower synonym scores), "
        "or 'none' to disable prompts. Avoid 'Clustering' — it groups by theme "
        "not entity identity.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for embedding.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2048,
        help="Rows per chunk for similarity matrix.",
    )
    parser.add_argument(
        "--no-louvain", action="store_true",
        help="Skip community detection.",
    )
    parser.add_argument(
        "--reuse-embeddings", action="store_true",
        help="Skip re-embedding if node_embeddings.npy + nodes.txt already exist in output-dir.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading triplets...", flush=True)
    triplets, nodes_from_triplets = load_all_triplets(args.triplet_dirs)
    print(
        f"  {len(triplets):,} triplets, {len(nodes_from_triplets):,} unique nodes",
        flush=True,
    )

    # Compute raw degree (for canonical selection)
    degrees: Counter = Counter()
    for t in triplets:
        for k in ("node_1", "node_2"):
            n = str(t.get(k, "")).lower().strip()
            if n:
                degrees[n] += 1

    emb_path = args.output_dir / "node_embeddings.npy"
    nodes_path = args.output_dir / "nodes.txt"

    if args.reuse_embeddings and emb_path.exists() and nodes_path.exists():
        print(f"\nReusing cached embeddings from {emb_path}", flush=True)
        t0 = time.time()
        emb = np.load(emb_path)
        cached_nodes = nodes_path.read_text().splitlines()
        if emb.shape[0] != len(cached_nodes):
            sys.exit(
                f"Cached embeddings shape {emb.shape} doesn't match "
                f"nodes.txt ({len(cached_nodes)} lines). Re-run without --reuse-embeddings."
            )
        print(
            f"  loaded {emb.shape[0]:,} embeddings (dim {emb.shape[1]}) "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )
        # Apply pathological-node filter and subset embeddings to match
        t0 = time.time()
        nodes, mask = clean_nodes(cached_nodes)
        print(f"  clean_nodes: {time.time() - t0:.1f}s", flush=True)
        t0 = time.time()
        emb = np.ascontiguousarray(emb[mask])
        print(
            f"  filtered out {len(cached_nodes) - len(nodes):,} pathological nodes "
            f"(math, quantities, ids); {len(nodes):,} remain "
            f"(subset {time.time() - t0:.1f}s)",
            flush=True,
        )
    else:
        # Apply filter before embedding so we don't waste compute
        nodes, _mask = clean_nodes(nodes_from_triplets)
        print(
            f"\nFiltered out {len(nodes_from_triplets) - len(nodes):,} pathological "
            f"nodes (math, quantities, ids); {len(nodes):,} remain",
            flush=True,
        )
        print(f"\nLoading model: {args.model}", flush=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  device: {device}", flush=True)
        model = SentenceTransformer(args.model, device=device)

        prompt_name = None if args.prompt.lower() == "none" else args.prompt
        emb = embed_nodes(model, nodes, args.batch_size, prompt_name=prompt_name)

        # Save embeddings for later reuse (post-filter)
        np.save(emb_path, emb)
        nodes_path.write_text("\n".join(nodes))
        print(f"Saved embeddings to {emb_path}", flush=True)

    canonical, clusters_named = cluster_and_canonicalize(
        emb,
        nodes,
        degrees,
        threshold=args.threshold,
        batch_size=args.chunk_size,
        output_dir=args.output_dir,
    )
    del emb  # cluster_and_canonicalize already freed the GPU copy

    print(f"\nClusters with merges: {len(clusters_named):,}", flush=True)
    print(f"Nodes remapped     : {len(canonical):,}", flush=True)

    # Save canonical mapping
    with open(args.output_dir / "canonical_map.tsv", "w") as f:
        f.write("original\tcanonical\n")
        for orig, canon in sorted(canonical.items()):
            f.write(f"{orig}\t{canon}\n")

    # Build deduplicated graph
    print("\nBuilding deduplicated graph...", flush=True)
    G = build_graph(triplets, canonical)
    print(f"  nodes: {G.number_of_nodes():,}", flush=True)
    print(f"  edges: {G.number_of_edges():,}", flush=True)

    # Community detection
    if not args.no_louvain:
        partition = run_louvain(G)
        if partition:
            for n, c in partition.items():
                G.nodes[n]["community"] = c

    # Save
    graphml_path = args.output_dir / "knowledge_graph_dedup.graphml"
    nx.write_graphml(G, str(graphml_path))
    print(f"\nSaved to {graphml_path}", flush=True)

    # Print the largest 20 clusters for sanity check
    print("\n--- Sample merges (top 20 largest clusters) ---", flush=True)
    top_clusters = sorted(clusters_named, key=len, reverse=True)[:20]
    for members in top_clusters:
        canon = canonical.get(members[0], members[0])
        # Show canon then up to 6 other members
        others = [m for m in members if m != canon][:6]
        more = f" (+{len(members) - 1 - len(others)} more)" if len(members) - 1 > len(others) else ""
        print(f"  [{len(members)}] {canon!r} <- {others}{more}", flush=True)


if __name__ == "__main__":
    main()
