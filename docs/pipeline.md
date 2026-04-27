# Pipeline Walkthrough

Step-by-step commands and expected output at each stage. Assumes you have
completed the [setup](../README.md#setup) and filled in `.env`.

## Stage 1 — Metadata prep

**Input:** A Web of Science search result exported as `.xls`, `.xlsx`, `.csv`,
or `.tsv`. To get one:

1. Run your query on <https://www.webofscience.com>
2. Click **Export → Excel** (or CSV)
3. Choose "Records from 1 to 1000" and a custom selection of fields that
   includes at least DOI, Article Title, Authors, Source Title, and
   Publication Year

**Command:**

```bash
python dataset_kg/01_prep_metadata.py \
    --input data/metadata/wos_export.xls \
    --output data/metadata/corpus.csv
```

**Output:** `data/metadata/corpus.csv` with an added `publisher` column. The
script prints a summary of publishers:

```
--- Publisher breakdown ---
  Elsevier: 894
  Springer: 179
  Wiley: 165
  MDPI: 237
  ...
  TOTAL: 1,891

Wiley + Elsevier + Springer: 1,238 (65%)
```

If you already have a previous corpus and want to deduplicate, add
`--existing data/metadata/previous_corpus.csv`.

## Stage 2 — PDF download

Tries publisher TDM APIs first, then Unpaywall as a fallback. A 5-second
delay between papers respects rate limits. **Before running at scale**,
read [`api_setup.md`](api_setup.md) to make sure your usage complies with
each publisher's terms.

**Command:**

```bash
python dataset_kg/02_download_pdfs.py \
    --metadata data/metadata/corpus.csv \
    --output-dir data/pdfs
```

**Output:** One PDF per DOI named `10.xxxx_paperid.pdf` in `data/pdfs/`,
plus a `download_log.csv` recording success/failure per DOI.

**To retry failures** after fixing access issues:

```bash
python dataset_kg/02_download_pdfs.py \
    --metadata data/metadata/corpus.csv \
    --output-dir data/pdfs \
    --retry-failed
```

## Stage 3 — PDF to markdown (Marker OCR)

Marker uses Surya OCR on GPU to convert each PDF into a structured markdown
file. It handles multi-column layout, tables, equations, figures, and scanned
documents.

**Command:**

```bash
python dataset_kg/03_pdf_to_markdown.py \
    --pdf-dir data/pdfs \
    --output-dir data/markdown
```

**Output:** For each `paper.pdf`, a subfolder `data/markdown/paper/` with:

* `paper.md` — the extracted markdown
* `paper_meta.json` — structural metadata
* `_page_N_Figure_M.jpeg` — extracted images (figures and decorative)

A CUDA GPU is strongly recommended; CPU works but is much slower.

**For PDFs where Marker fails** (output is tiny or empty):

```bash
python dataset_kg/03_pdf_to_markdown.py \
    --pdf-dir data/pdfs \
    --output-dir data/markdown \
    --force-ocr
```

`--force-ocr` treats every page as a scanned image, which helps with
encrypted, scanned, or heavily image-based PDFs.

## Stage 4 — Trim markdown

Strips the title page, journal headers, references, acknowledgements, and
decorative images — keeping just the paper title plus the scientific content
from Abstract through Conclusions.

**Command:**

```bash
python dataset_kg/04_trim_markdown.py --input-dir data/markdown
```

Add `--dry-run` first to preview what will be removed.

**Effect:** Typically ~35% reduction in line count and deletion of
decorative images (journal logos, author photos, QR codes). Actual figures
with captions are preserved.

## Stage 5 — Knowledge graph extraction

For each markdown paper, the
[`graphPrompt`](https://github.com/lamm-mit/GraphReasoning) pipeline
(LAMM lab, MIT) sends every 2,500-char chunk through three LLM calls:

1. Extract ~10 `(node, edge, node)` triplets
2. Standardize node names to consistent terminology for the field
   (the prompt currently asks for materials-science conventions; edit
   `SYS_PROMPT_FORMAT` in `dataset_kg/05_extract_kg.py` for other domains)
3. Validate JSON format

**Command:**

```bash
python dataset_kg/05_extract_kg.py \
    --input-dir data/markdown \
    --output-dir data/kg
```

**Output:**

* `data/kg/triplets/<paper>.json` — one JSON array per paper containing its
  extracted triplets
* `data/kg/knowledge_graph.graphml` — a merged NetworkX-compatible graph
  file readable by Gephi, Cytoscape, or Python tools

**Test on a few papers before the full run:**

```bash
python dataset_kg/05_extract_kg.py \
    --input-dir data/markdown \
    --output-dir data/kg \
    --limit 5
```

This processes 5 papers and lets you see the actual triplet output before
committing to the full corpus.

## Stage 6 — Embedding-based node deduplication

Raw triplets contain many synonyms — "alumina" / "Al2O3" / "aluminum oxide"
appear as three separate nodes that should be one. This stage merges them
using EmbeddingGemma-300m embeddings and `community_detection` from
sentence-transformers.

**Command:**

```bash
python dataset_kg/06_dedup_kg.py \
    --triplet-dirs data/kg/triplets \
    --output-dir data/kg/dedup
```

To merge multiple corpora in one go (e.g. bio-inspired + ceramic-AM), pass
multiple `--triplet-dirs` arguments:

```bash
python dataset_kg/06_dedup_kg.py \
    --triplet-dirs data/kg/bioinspired/triplets \
                   data/kg/ceramic_am/triplets \
    --output-dir data/kg/dedup
```

**Algorithm:**

1. Filter pathological node names (LaTeX math, pure quantities like "2 mm",
   single characters).
2. Embed every node with EmbeddingGemma using the **STS** (Semantic Textual
   Similarity) prompt.
3. Find clusters of near-duplicates with chunked GPU cosine via
   `sentence_transformers.util.community_detection` at `threshold=0.88`.
4. Pick a canonical name per cluster (highest in-graph degree wins;
   shortest label as tiebreaker).
5. Rebuild the graph with merged nodes and run Louvain community detection.

**Output:**

* `data/kg/dedup/node_embeddings.npy` — cached embeddings (~1.3 GB / 400K
  nodes); reusable via `--reuse-embeddings`
* `data/kg/dedup/nodes.txt` — node-name list aligned with the embeddings
* `data/kg/dedup/merge_clusters.tsv` — every merged cluster with its
  members; **inspect this** to verify quality
* `data/kg/dedup/canonical_map.tsv` — one row per remapped node:
  `original_name → canonical_name`
* `data/kg/dedup/knowledge_graph_dedup.graphml` — final graph with
  Louvain `community` attribute on each node

**Tuning:**

* `--threshold` (default 0.88). Raise to 0.90 if your domain needs
  stricter "must be near-identical" merging; lower to 0.85 to catch
  more typographically-distant variants at the cost of broader concepts
  occasionally absorbing specific ones.
* `--prompt` (default `STS`). Try `Retrieval-document` for cleaner
  unrelated/synonym separation at the cost of slightly weaker synonym
  matching. **Avoid `Clustering` — it groups by theme, not identity.**
* `--reuse-embeddings` skips the 8-min embed step when iterating on
  threshold or prompt.

