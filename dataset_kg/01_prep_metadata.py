"""
01_prep_metadata.py — Parse a Web of Science (WoS) export, deduplicate against an
existing corpus if provided, and classify each DOI by publisher.

The output is a CSV suitable as input to 02_download_pdfs.py.

Usage:
    # Basic — new corpus from a fresh WoS export
    python dataset_kg/01_prep_metadata.py \\
        --input data/metadata/wos_export.xls \\
        --output data/metadata/corpus.csv

    # Deduplicate against an existing corpus
    python dataset_kg/01_prep_metadata.py \\
        --input data/metadata/wos_export.csv \\
        --existing data/metadata/existing_corpus.csv \\
        --output data/metadata/new_papers.csv

    # Skip CrossRef lookup for faster prep (DOI prefix classification only)
    python dataset_kg/01_prep_metadata.py \\
        --input data/metadata/wos_export.csv \\
        --output data/metadata/corpus.csv \\
        --no-crossref
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

load_dotenv()

USER_AGENT = os.getenv("USER_AGENT", "CorpusBuilder/1.0")

# DOI prefix → publisher mapping (covers most major publishers).
# Used as a fast first pass before optionally querying CrossRef.
DOI_PREFIX_MAP: dict[str, str] = {
    "10.1002": "Wiley",
    "10.1046": "Wiley",
    "10.1111": "Wiley",
    "10.1006": "Elsevier",
    "10.1016": "Elsevier",
    "10.1007": "Springer",
    "10.1023": "Springer",
    "10.1038": "Springer",   # Nature Portfolio
    "10.1057": "Springer",
    "10.1140": "Springer",
    "10.1186": "Springer",   # BioMed Central
    "10.1251": "Springer",
    "10.1361": "Springer",   # ASM International
    "10.1557": "Springer",   # MRS
    "10.1021": "ACS",
    "10.1039": "RSC",
    "10.1063": "AIP",
    "10.1080": "Taylor & Francis",
    "10.1088": "IOP",
    "10.1103": "APS",
    "10.1126": "Science/AAAS",
    "10.1073": "PNAS",
    "10.1177": "SAGE",
    "10.1108": "Emerald",
    "10.3390": "MDPI",
    "10.3791": "JoVE",
    "10.4028": "Trans Tech",
}


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------

def classify_by_prefix(doi: str) -> str:
    """Classify publisher from DOI prefix. Returns 'Other' if no match."""
    if not doi:
        return "Unknown"
    for prefix, publisher in DOI_PREFIX_MAP.items():
        if doi.startswith(prefix):
            return publisher
    return "Other"


def classify_via_crossref(doi: str) -> str | None:
    """Query CrossRef for publisher name. Returns None on failure."""
    try:
        resp = requests.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        publisher = (
            resp.json().get("message", {}).get("publisher", "").strip()
        )
        if not publisher:
            return None
        low = publisher.lower()
        if "wiley" in low:
            return "Wiley"
        if "elsevier" in low or "sciencedirect" in low:
            return "Elsevier"
        if "springer" in low or "nature" in low:
            return "Springer"
        return publisher
    except Exception:
        return None


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------

def read_wos_export(path: Path) -> pd.DataFrame:
    """Read a WoS export in CSV, TSV, XLS, or XLSX format."""
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    # Try common CSV encodings first
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    # Fall back to tab-separated
    return pd.read_csv(path, sep="\t", encoding="utf-8-sig")


def find_doi_column(df: pd.DataFrame) -> str:
    """Locate the DOI column in a WoS DataFrame."""
    for col in df.columns:
        if col.strip().upper() in {"DOI", "DI", "D2"}:
            return col
    raise ValueError(
        f"No DOI column found. Columns available: {list(df.columns)}"
    )


def load_existing_dois(path: Path) -> set[str]:
    """Load DOIs from an existing corpus CSV for deduplication."""
    if not path.exists():
        raise FileNotFoundError(f"Existing corpus file not found: {path}")
    df = pd.read_csv(path)
    doi_col = find_doi_column(df)
    return set(df[doi_col].dropna().astype(str).str.strip().str.lower())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prep a Web of Science metadata export for corpus building.",
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Path to WoS export (CSV, TSV, XLS, or XLSX).",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Path to write the deduplicated + classified CSV.",
    )
    parser.add_argument(
        "--existing", type=Path, default=None,
        help="Optional existing corpus CSV for deduplication.",
    )
    parser.add_argument(
        "--no-crossref", action="store_true",
        help="Skip CrossRef lookups for 'Other' publishers (faster).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load the WoS export
    df = read_wos_export(args.input)
    print(f"Loaded {len(df)} records from {args.input.name}")

    # Locate the DOI column
    try:
        doi_col = find_doi_column(df)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    df["doi_clean"] = (
        df[doi_col].fillna("").astype(str).str.strip().str.lower()
    )

    # Drop rows without a DOI
    missing = df["doi_clean"].eq("")
    if missing.any():
        print(f"Warning: {missing.sum()} records have no DOI (dropped)")
        df = df[~missing]

    # Deduplicate against existing corpus
    if args.existing:
        existing = load_existing_dois(args.existing)
        print(f"Existing corpus: {len(existing)} DOIs")
        is_dup = df["doi_clean"].isin(existing)
        n_dup = int(is_dup.sum())
        df = df[~is_dup]
        print(f"Removed {n_dup} duplicates, {len(df)} new papers remaining")

    # Classify by publisher (prefix)
    print("Classifying publishers by DOI prefix...")
    df["publisher"] = df["doi_clean"].apply(classify_by_prefix)

    # Refine 'Other' via CrossRef
    if not args.no_crossref:
        mask_other = df["publisher"].eq("Other")
        if mask_other.any():
            print(f"Querying CrossRef for {int(mask_other.sum())} 'Other' DOIs...")
            for idx in df[mask_other].index:
                doi = df.at[idx, "doi_clean"]
                pub = classify_via_crossref(doi)
                if pub:
                    df.at[idx, "publisher"] = pub
                time.sleep(0.5)  # be polite

    df = df.drop(columns=["doi_clean"])

    # Write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} records to {args.output}")

    # Summary
    counts = df["publisher"].value_counts()
    print("\n--- Publisher breakdown ---")
    for pub, count in counts.items():
        print(f"  {pub}: {count}")

    big3 = int(
        counts.get("Wiley", 0) + counts.get("Elsevier", 0) + counts.get("Springer", 0)
    )
    pct = big3 / len(df) * 100 if len(df) else 0
    print(f"\nWiley + Elsevier + Springer: {big3} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
