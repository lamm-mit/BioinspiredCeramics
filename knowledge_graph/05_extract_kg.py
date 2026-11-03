"""
05_extract_kg.py — Extract a knowledge graph of scientific concepts and their
relationships from a corpus of markdown papers.

Implementation follows the graphPrompt pipeline from GraphReasoning
(https://github.com/lamm-mit/GraphReasoning). Each paper's markdown is chunked
into ~2,500-character windows. Every chunk is sent through three LLM calls:

  1. Triplet extraction  — ~10 (node, edge, node) relationships per chunk
  2. Standardization     — normalize node names to materials-science terms
  3. Format validation   — ensure valid JSON output

Output is one JSON file per paper containing all extracted triplets. The files
together describe a knowledge graph of the corpus, suitable for downstream
analysis (community detection, embedding-based deduplication, path finding).

Usage:
    # Extract triplets for every markdown paper
    python knowledge_graph/05_extract_kg.py \\
        --input-dir data/markdown \\
        --output-dir data/kg

    # Test on 5 papers first
    python knowledge_graph/05_extract_kg.py \\
        --input-dir data/markdown \\
        --output-dir data/kg \\
        --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import networkx as nx
import openai
from dotenv import load_dotenv


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

load_dotenv()

CHUNK_SIZE = 2500
CHUNK_OVERLAP = 100
DEFAULT_MODEL = "gpt-4o-mini"


# graphPrompt system prompts (from lamm-mit/GraphReasoning)
SYS_PROMPT_GRAPHMAKER = """\
You are a network ontology graph maker who extracts terms and their relations \
from a given context, using category theory. You are provided with a context \
chunk (delimited by ```). Your task is to extract the ontology of terms \
mentioned in the given context. These terms should represent the key concepts \
as per the context, including well-defined and widely used names of materials, \
systems, methods.

Format your output as a list of JSON. Each element of the list contains a pair \
of terms and the relation between them, like the following:
[
   {"node_1": "A concept from extracted ontology", \
"node_2": "A related concept from extracted ontology", \
"edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"}, \
{...}
]

Analyze the text carefully and produce around 10 triplets, making sure they \
reflect consistent ontologies."""

SYS_PROMPT_FORMAT = """\
You respond in this format:
[{"node_1": "A concept from extracted ontology", \
"node_2": "A related concept from extracted ontology", \
"edge": "Relationship between the two concepts, node_1 and node_2, succinctly described"}, \
{...}]"""


# ----------------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-bounded chunks."""
    chunks: list[str] = []
    step = size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i : i + size])
    return chunks


def extract_json(s: str) -> str:
    """Extract a JSON array from arbitrary LLM output by finding first [ and last ]."""
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1:
        return "[]"
    return s[start : end + 1]


def call_llm(
    client: openai.OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """One chat completion call. Raises on API errors."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def graph_prompt(text: str, client: openai.OpenAI, model: str) -> list[dict]:
    """Run the three-call graphPrompt pipeline on one text chunk."""
    try:
        # Call 1 — extract triplets
        raw = call_llm(
            client, model,
            SYS_PROMPT_GRAPHMAKER,
            f"Context: ```{text}``` \n\nOutput: ",
        )
        # Call 2 — standardize terminology
        raw = call_llm(
            client, model,
            SYS_PROMPT_FORMAT,
            (
                f"Read this context: ```{text}```.\n"
                f"Read this ontology: ```{raw}```\n\n"
                "Improve the ontology by renaming nodes so that they have "
                "consistent labels that are widely used in the field of "
                "materials science."
            ),
        )
        # Call 3 — validate/fix JSON format
        raw = call_llm(
            client, model,
            SYS_PROMPT_FORMAT,
            f"Context: ```{raw}``` \n\n Fix to make sure it is proper format.",
        )
        raw = raw.replace("\\", "")
        return json.loads(extract_json(raw))
    except Exception:
        return []


# ----------------------------------------------------------------------------
# Per-paper processing
# ----------------------------------------------------------------------------

def process_paper(
    paper_dir: Path,
    output_dir: Path,
    client: openai.OpenAI,
    model: str,
) -> tuple[str, int, str]:
    """Extract triplets for one paper. Returns (paper_name, triplet_count, status)."""
    name = paper_dir.name
    result_file = output_dir / f"{name}.json"

    if result_file.exists():
        return name, 0, "skipped"

    md_files = list(paper_dir.glob("*.md"))
    if not md_files:
        return name, 0, "no markdown"

    text = md_files[0].read_text(encoding="utf-8", errors="replace")
    if len(text) < 100:
        return name, 0, "too short"

    chunks = chunk_text(text)
    triplets: list[dict] = []

    for chunk in chunks:
        extracted = graph_prompt(chunk, client, model)
        for t in extracted:
            t["paper"] = name
        triplets.extend(extracted)

    result_file.write_text(json.dumps(triplets))
    return name, len(triplets), "ok"


# ----------------------------------------------------------------------------
# Graph assembly
# ----------------------------------------------------------------------------

def merge_triplets_to_graph(triplets_dir: Path, output_dir: Path) -> None:
    """Merge all per-paper triplet JSONs into one NetworkX graph."""
    print(f"\nMerging triplets from {triplets_dir}...")

    G = nx.DiGraph()
    total = 0

    for tf in sorted(triplets_dir.glob("*.json")):
        try:
            triplets = json.loads(tf.read_text())
        except Exception as e:
            print(f"  skip {tf.name}: {e}")
            continue

        for t in triplets:
            n1 = str(t.get("node_1", "")).lower().strip()
            n2 = str(t.get("node_2", "")).lower().strip()
            edge = str(t.get("edge", "")).strip()
            if n1 and n2:
                G.add_edge(n1, n2, title=edge, paper=t.get("paper", ""))
                total += 1

    graphml_path = output_dir / "knowledge_graph.graphml"
    nx.write_graphml(G, str(graphml_path))
    print(f"  nodes  : {G.number_of_nodes():,}")
    print(f"  edges  : {G.number_of_edges():,}")
    print(f"  triplets total: {total:,}")
    print(f"  saved  : {graphml_path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a knowledge graph from markdown papers via graphPrompt.",
    )
    parser.add_argument(
        "--input-dir", "-i", required=True, type=Path,
        help="Directory of per-paper subfolders, each containing <name>.md",
    )
    parser.add_argument(
        "--output-dir", "-o", required=True, type=Path,
        help="Directory to write per-paper triplet JSON files and the merged graph.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"OpenAI model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum papers to process this run (for testing).",
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Skip merging per-paper triplets into a NetworkX graph.",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment / .env", file=sys.stderr)
        sys.exit(1)

    if not args.input_dir.exists():
        print(f"Error: input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    triplets_dir = args.output_dir / "triplets"
    triplets_dir.mkdir(parents=True, exist_ok=True)

    paper_dirs = sorted([p for p in args.input_dir.iterdir() if p.is_dir()])
    if args.limit:
        paper_dirs = paper_dirs[: args.limit]

    already_done = sum(
        1 for p in paper_dirs if (triplets_dir / f"{p.name}.json").exists()
    )
    print(
        f"Papers: {len(paper_dirs)} total, {already_done} already extracted, "
        f"{len(paper_dirs) - already_done} to do"
    )
    print(f"Model  : {args.model}\n")

    client = openai.OpenAI(api_key=api_key)

    total_triplets = 0
    done = 0
    failed = 0
    for i, paper_dir in enumerate(paper_dirs, 1):
        name, n_triplets, status = process_paper(
            paper_dir, triplets_dir, client, args.model
        )
        if status == "ok":
            total_triplets += n_triplets
            done += 1
            print(f"  [{i}/{len(paper_dirs)}] {name}: {n_triplets} triplets")
        elif status == "skipped":
            pass
        else:
            failed += 1
            print(f"  [{i}/{len(paper_dirs)}] {name}: {status}")

    print("\n--- Summary ---")
    print(f"Papers done    : {done}")
    print(f"Papers failed  : {failed}")
    print(f"Papers skipped : {already_done}")
    print(f"Total triplets : {total_triplets}")

    if not args.no_merge:
        merge_triplets_to_graph(triplets_dir, args.output_dir)


if __name__ == "__main__":
    main()
