"""
03_pdf_to_markdown.py — Batch-convert PDFs to markdown using Marker.

Marker uses GPU-accelerated OCR (Surya) and handles tables, figures, equations,
and scanned PDFs far better than text-only extractors like pdftotext.

Requires: pip install marker-pdf  (PyTorch; GPU strongly recommended)

Usage:
    # Convert everything in the PDF directory
    python dataset_kg/03_pdf_to_markdown.py \\
        --pdf-dir data/pdfs \\
        --output-dir data/markdown

    # Process up to 10 papers
    python dataset_kg/03_pdf_to_markdown.py \\
        --pdf-dir data/pdfs \\
        --output-dir data/markdown \\
        --limit 10

    # Retry PDFs that produced empty/tiny output, forcing OCR
    python dataset_kg/03_pdf_to_markdown.py \\
        --pdf-dir data/pdfs \\
        --output-dir data/markdown \\
        --force-ocr

Notes:
    * Marker writes one folder per PDF, containing `<stem>.md` plus extracted
      images and a `<stem>_meta.json` metadata file.
    * Sequential processing only — Marker's GPU model loading + per-PDF
      memory footprint makes parallel workers unreliable in practice.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TIMEOUT_SECONDS = 600  # 10 minutes per PDF


def expected_output(pdf_path: Path, markdown_dir: Path) -> Path:
    """Return the expected markdown file path for a given input PDF."""
    return markdown_dir / pdf_path.stem / f"{pdf_path.stem}.md"


def convert_one(
    pdf_path: Path,
    markdown_dir: Path,
    force_ocr: bool,
    timeout: int,
) -> tuple[str, bool, str]:
    """Convert a single PDF using the `marker_single` CLI."""
    md_path = expected_output(pdf_path, markdown_dir)

    if md_path.exists() and md_path.stat().st_size > 0 and not force_ocr:
        return pdf_path.name, True, "already exists"

    cmd = [
        "marker_single",
        str(pdf_path),
        "--output_dir", str(markdown_dir),
        "--output_format", "markdown",
    ]
    if force_ocr:
        cmd.append("--force_ocr")

    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=timeout
        )
        if md_path.exists():
            return pdf_path.name, True, "converted"
        return pdf_path.name, False, "marker ran but no output file"
    except subprocess.TimeoutExpired:
        return pdf_path.name, False, f"timeout (>{timeout}s)"
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or "").strip().splitlines()[-1] if e.stderr else str(e)
        return pdf_path.name, False, f"marker error: {msg[:200]}"
    except FileNotFoundError:
        return (
            pdf_path.name,
            False,
            "marker_single not found — install with `pip install marker-pdf`",
        )
    except Exception as e:
        return pdf_path.name, False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-convert PDFs to markdown via Marker (OCR).",
    )
    parser.add_argument(
        "--pdf-dir", "-i", required=True, type=Path,
        help="Directory containing input PDFs.",
    )
    parser.add_argument(
        "--output-dir", "-o", required=True, type=Path,
        help="Directory for markdown output (one subfolder per PDF).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of PDFs to process this run.",
    )
    parser.add_argument(
        "--force-ocr", action="store_true",
        help="Force OCR on every page (useful for scanned or image-based PDFs).",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout per PDF in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    args = parser.parse_args()

    if not args.pdf_dir.exists():
        print(f"Error: PDF directory not found: {args.pdf_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    if not pdfs:
        print(f"No PDFs found in {args.pdf_dir}", file=sys.stderr)
        sys.exit(0)

    print(
        f"Processing {len(pdfs)} PDFs"
        f"{' (force OCR)' if args.force_ocr else ''}..."
    )

    ok = fail = skip = 0
    for pdf in pdfs:
        name, success, msg = convert_one(
            pdf, args.output_dir, args.force_ocr, args.timeout
        )
        if success and msg == "already exists":
            skip += 1
        elif success:
            ok += 1
            print(f"  OK    {name}")
        else:
            fail += 1
            print(f"  FAIL  {name}  [{msg}]")

    print("\n--- Summary ---")
    print(f"Converted : {ok}")
    print(f"Skipped   : {skip}")
    print(f"Failed    : {fail}")
    print(f"Output    : {args.output_dir}")


if __name__ == "__main__":
    main()
