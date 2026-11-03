"""
02_download_pdfs.py — Download full-text PDFs for a list of DOIs.

Tries publisher-sanctioned text-and-data-mining (TDM) routes only:
  1. Wiley TDM API            (Wiley DOIs)
  2. Elsevier full-text API   (Elsevier DOIs)
  3. Springer Nature APIs     (Open Access API, then Meta API)
  4. ACS direct PDF endpoint  (IP-authenticated institutional subscription)
  5. Unpaywall fallback       (openly-licensed copies for everything else)

This script does not bypass paywalls, spoof user-agents, or evade rate
limits. The default rate limit is one request every 5 seconds. You are
responsible for ensuring your usage complies with each publisher's API
terms and your institution's TDM agreement — see knowledge_graph/api_setup.md for
the legal framing.

Reads API keys and a contact email from .env (see .env.example).

Usage:
    # Download everything in the metadata CSV
    python knowledge_graph/02_download_pdfs.py \\
        --metadata data/metadata/corpus.csv \\
        --output-dir data/pdfs

    # Try only a subset
    python knowledge_graph/02_download_pdfs.py \\
        --metadata data/metadata/corpus.csv \\
        --output-dir data/pdfs \\
        --limit 50

    # Retry previously failed DOIs
    python knowledge_graph/02_download_pdfs.py \\
        --metadata data/metadata/corpus.csv \\
        --output-dir data/pdfs \\
        --retry-failed
"""

from __future__ import annotations

import argparse
import csv
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

WILEY_TDM_TOKEN = os.getenv("WILEY_TDM_TOKEN", "")
ELSEVIER_API_KEY = os.getenv("ELSEVIER_API_KEY", "")
SPRINGER_OA_API_KEY = os.getenv("SPRINGER_OA_API_KEY", "")
SPRINGER_META_API_KEY = os.getenv("SPRINGER_META_API_KEY", "")

CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "contact@example.org")
USER_AGENT = os.getenv(
    "USER_AGENT",
    f"CorpusBuilder/1.0 (mailto:{CONTACT_EMAIL})",
)

RATE_LIMIT_SECONDS = 5     # Delay between DOIs (respect publisher rate limits)
REQUEST_TIMEOUT = 30       # Per-request HTTP timeout


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def doi_to_filename(doi: str) -> str:
    """Convert a DOI to a filesystem-safe PDF filename."""
    return doi.replace("/", "_") + ".pdf"


def is_valid_pdf(content: bytes) -> bool:
    """Check if byte content looks like a real PDF."""
    return bool(content and len(content) >= 1000 and content[:5].startswith(b"%PDF"))


def save_pdf(content: bytes, doi: str, pdf_dir: Path) -> bool:
    """Write PDF bytes to disk if they're valid. Returns True on success."""
    if not is_valid_pdf(content):
        return False
    (pdf_dir / doi_to_filename(doi)).write_bytes(content)
    return True


def classify_publisher(name: str) -> str:
    """Normalize a publisher string to a canonical key used by the dispatcher."""
    low = str(name).lower()
    if "wiley" in low:
        return "Wiley"
    if "elsevier" in low or "pergamon" in low or "cell press" in low:
        return "Elsevier"
    if "springer" in low or "nature portfolio" in low:
        return "Springer"
    if "amer chemical" in low or "american chemical" in low or low == "acs":
        return "ACS"
    return "Other"


def load_download_log(log_path: Path) -> dict[str, str]:
    """Load a previous download log as {lowercased_doi: status}."""
    if not log_path.exists():
        return {}
    log: dict[str, str] = {}
    with open(log_path) as f:
        for row in csv.DictReader(f):
            log[row["doi"]] = row["status"]
    return log


def append_log(
    log_path: Path, doi: str, publisher: str, status: str, note: str = ""
) -> None:
    """Append a single row to the download log."""
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["doi", "publisher", "status", "note"])
        writer.writerow([doi, publisher, status, note])


# ----------------------------------------------------------------------------
# Publisher-specific downloaders
# ----------------------------------------------------------------------------

def download_wiley(doi: str, pdf_dir: Path) -> tuple[bool, str]:
    if not WILEY_TDM_TOKEN:
        return False, "WILEY_TDM_TOKEN not set"
    try:
        resp = requests.get(
            f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}",
            headers={
                "Wiley-TDM-Client-Token": WILEY_TDM_TOKEN,
                "User-Agent": USER_AGENT,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200 and save_pdf(resp.content, doi, pdf_dir):
            return True, "OK"
        return False, f"Wiley TDM {resp.status_code}"
    except Exception as e:
        return False, f"Wiley TDM error: {e}"


def download_elsevier(doi: str, pdf_dir: Path) -> tuple[bool, str]:
    if not ELSEVIER_API_KEY:
        return False, "ELSEVIER_API_KEY not set"
    try:
        resp = requests.get(
            f"https://api.elsevier.com/content/article/doi/{doi}",
            headers={
                "X-ELS-APIKey": ELSEVIER_API_KEY,
                "Accept": "application/pdf",
                "User-Agent": USER_AGENT,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200 and save_pdf(resp.content, doi, pdf_dir):
            return True, "OK"
        return False, f"Elsevier {resp.status_code}"
    except Exception as e:
        return False, f"Elsevier error: {e}"


def download_acs(doi: str, pdf_dir: Path) -> tuple[bool, str]:
    """ACS direct PDF. Requires institutional IP access."""
    try:
        resp = requests.get(
            f"https://pubs.acs.org/doi/pdf/{doi}",
            headers={"Accept": "application/pdf", "User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        if resp.status_code == 200 and save_pdf(resp.content, doi, pdf_dir):
            return True, "OK"
        return False, f"ACS {resp.status_code}"
    except Exception as e:
        return False, f"ACS error: {e}"


def download_springer(doi: str, pdf_dir: Path) -> tuple[bool, str]:
    """Springer Nature: try Open Access API first, then Meta API."""
    if not (SPRINGER_OA_API_KEY or SPRINGER_META_API_KEY):
        return False, "No Springer API keys set"

    # Open Access API — returns full text for OA papers
    if SPRINGER_OA_API_KEY:
        try:
            resp = requests.get(
                "https://api.springernature.com/openaccess/json",
                params={"q": f"doi:{doi}", "api_key": SPRINGER_OA_API_KEY},
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                for rec in resp.json().get("records", []):
                    for u in rec.get("url", []):
                        if u.get("format") == "pdf":
                            time.sleep(2)
                            r = requests.get(
                                u["value"],
                                headers={"User-Agent": USER_AGENT},
                                timeout=REQUEST_TIMEOUT,
                            )
                            if (
                                r.status_code == 200
                                and save_pdf(r.content, doi, pdf_dir)
                            ):
                                return True, "OK (Open Access)"
        except Exception:
            pass

    # Meta API — metadata + PDF links for subscription content
    if SPRINGER_META_API_KEY:
        time.sleep(RATE_LIMIT_SECONDS)
        try:
            resp = requests.get(
                "https://api.springernature.com/meta/v2/json",
                params={"q": f"doi:{doi}", "api_key": SPRINGER_META_API_KEY},
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                return False, f"Springer Meta {resp.status_code}"

            pdf_url = None
            for rec in resp.json().get("records", []):
                for u in rec.get("url", []):
                    if u.get("format") == "pdf":
                        pdf_url = u["value"]
                        break
                if pdf_url:
                    break

            if not pdf_url:
                return False, "No PDF URL in Springer Meta"

            time.sleep(RATE_LIMIT_SECONDS)
            resp = requests.get(
                pdf_url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200 and save_pdf(resp.content, doi, pdf_dir):
                return True, "OK (Meta)"
            return False, f"Springer PDF {resp.status_code}"
        except Exception as e:
            return False, f"Springer error: {e}"

    return False, "Springer APIs returned no PDF"


def download_unpaywall(doi: str, pdf_dir: Path) -> tuple[bool, str]:
    """Unpaywall fallback — tries to find an open-access copy anywhere on the web."""
    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": CONTACT_EMAIL},
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return False, f"Unpaywall {resp.status_code}"

        best_oa = resp.json().get("best_oa_location")
        if not best_oa:
            return False, "No OA location"

        pdf_url = best_oa.get("url_for_pdf")
        if not pdf_url:
            return False, "No PDF URL in OA location"

        time.sleep(2)
        resp = requests.get(
            pdf_url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
        )
        if resp.status_code == 200 and save_pdf(resp.content, doi, pdf_dir):
            return True, "OK (Unpaywall)"
        return False, f"OA PDF {resp.status_code}"
    except Exception as e:
        return False, f"Unpaywall error: {e}"


# ----------------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------------

DOWNLOADERS = {
    "Wiley": download_wiley,
    "Elsevier": download_elsevier,
    "Springer": download_springer,
    "ACS": download_acs,
}


def download_doi(doi: str, publisher_raw: str, pdf_dir: Path) -> tuple[bool, str]:
    """Try the publisher-specific route first, then fall back to Unpaywall."""
    publisher = classify_publisher(publisher_raw)
    downloader = DOWNLOADERS.get(publisher)
    if downloader:
        ok, note = downloader(doi, pdf_dir)
        if ok:
            return True, note
    time.sleep(RATE_LIMIT_SECONDS)
    return download_unpaywall(doi, pdf_dir)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PDFs for a list of DOIs from publisher APIs.",
    )
    parser.add_argument(
        "--metadata", "-m", required=True, type=Path,
        help="CSV with DOI column (output of 01_prep_metadata.py).",
    )
    parser.add_argument(
        "--output-dir", "-o", required=True, type=Path,
        help="Directory to save PDFs.",
    )
    parser.add_argument(
        "--log", type=Path, default=None,
        help="Download log CSV (default: <output_dir>/download_log.csv).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of papers to download this run.",
    )
    parser.add_argument(
        "--publisher", default=None,
        help="Only download papers whose Publisher column matches this string.",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Retry DOIs that failed in a previous run.",
    )
    args = parser.parse_args()

    if not args.metadata.exists():
        print(f"Error: metadata file not found: {args.metadata}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log or (args.output_dir / "download_log.csv")

    df = pd.read_csv(args.metadata)

    # Find DOI column
    doi_col = next(
        (c for c in df.columns if c.strip().upper() == "DOI"), None
    )
    if doi_col is None:
        print(f"Error: No DOI column in {args.metadata}", file=sys.stderr)
        sys.exit(1)

    if args.publisher:
        mask = df["publisher"].astype(str).str.contains(
            args.publisher, case=False, na=False
        )
        df = df[mask]
        print(f"Filtered to {len(df)} {args.publisher} papers")

    log = load_download_log(log_path)

    ok_count = 0
    fail_count = 0
    skip_count = 0
    processed = 0

    for _, row in df.iterrows():
        doi = str(row[doi_col]).strip()
        if not doi or doi == "nan":
            continue
        publisher_raw = str(row.get("publisher", "Other"))

        # Already downloaded — skip
        if (args.output_dir / doi_to_filename(doi)).exists():
            skip_count += 1
            continue

        # Logged previously — skip unless retrying
        key = doi.lower()
        if key in log:
            if log[key] == "success" or not args.retry_failed:
                skip_count += 1
                continue

        if args.limit and processed >= args.limit:
            break

        processed += 1
        print(f"[{processed}] {doi} ({publisher_raw})... ", end="", flush=True)

        ok, note = download_doi(doi, publisher_raw, args.output_dir)
        status = "success" if ok else "failed"
        append_log(log_path, key, publisher_raw, status, note)

        if ok:
            ok_count += 1
            print("OK")
        else:
            fail_count += 1
            print(f"FAILED ({note})")

        time.sleep(RATE_LIMIT_SECONDS)

    print("\n--- Summary ---")
    print(f"Downloaded : {ok_count}")
    print(f"Failed     : {fail_count}")
    print(f"Skipped    : {skip_count}")
    print(f"Log        : {log_path}")


if __name__ == "__main__":
    main()
