"""
04_trim_markdown.py — Post-process Marker output: strip front/back matter and
non-figure images from each paper's markdown.

Removes:
  * Everything before the Abstract (keeps the paper title heading only).
  * Everything after Conclusions — acknowledgements, references, funding,
    author contributions, data availability, etc.
  * Images without a captioned figure nearby (e.g. journal logos, headshots,
    decorative graphics). Only images whose next non-blank line begins with
    "Figure N", "Fig. N", "Scheme N", or "Table N" are kept.

Operates in-place on the markdown files and deletes uncaptioned image files.

Usage:
    # Trim in place
    python knowledge_graph/04_trim_markdown.py --input-dir data/markdown

    # Preview changes without modifying anything
    python knowledge_graph/04_trim_markdown.py --input-dir data/markdown --dry-run
"""

import argparse
import os
import re
import sys
from pathlib import Path


# Headings that mark the START of back matter (case-insensitive).
# Matches if the heading text starts with any of these.
BACK_MATTER_PATTERNS = [
    r"acknowledg",
    r"reference",
    r"bibliograph",
    r"funding",
    r"author\s+contribut",
    r"conflict.?\s+of\s+interest",
    r"supplementar",
    r"supporting\s+information",
    r"appendi",
    r"declaration",
    r"data\s+availab",
    r"credit\b",
    r"competing\s+interest",
    r"ethics\s+statement",
    r"disclosure",
]

# Regex to match any markdown heading line (# through ######)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)

# Regex to detect abstract heading or bold "Abstract" line
ABSTRACT_RE = re.compile(
    r"^#{1,6}\s+\**\s*abstract\s*\**\s*$|"
    r"^\**\s*abstract\s*\**\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def is_back_matter_heading(heading_text: str) -> bool:
    """Check if heading text matches a back-matter pattern."""
    clean = re.sub(r"[*_`#]", "", heading_text).strip()
    for pat in BACK_MATTER_PATTERNS:
        if re.match(pat, clean, re.IGNORECASE):
            return True
    return False


def find_title_end(lines: list[str]) -> int:
    """Find the line index after the title block.

    The title is the first markdown heading (# ...). We keep just that heading
    line and return the index after it. If no heading is found before the
    abstract, fall back to the first non-empty, non-image line.
    """
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Look for a markdown heading that isn't "Abstract"
        if re.match(r"^#{1,6}\s+", stripped):
            heading_text = re.sub(r"^#{1,6}\s+", "", stripped)
            heading_clean = re.sub(r"[*_`]", "", heading_text).strip().lower()
            if heading_clean in ("abstract",):
                # No title heading found before abstract — return 0
                return 0
            return i + 1
    return 0


def find_abstract_start(lines: list[str], title_end: int) -> int | None:
    """Find the line index where the abstract heading/marker appears."""
    for i in range(title_end, len(lines)):
        if ABSTRACT_RE.match(lines[i].strip()):
            return i
    # Fallback: look for "ABSTRACT" as a standalone word in any line
    for i in range(title_end, min(title_end + 50, len(lines))):
        if re.search(r"\babstract\b", lines[i], re.IGNORECASE):
            return i
    return None


# Regex to detect reference list entries (numbered citations)
# Matches: [1] Author, 1. Author, - [3] Author, <sup>[1]</sup> Author
REF_ENTRY_RE = re.compile(
    r"^\s*[-•]?\s*\[?\d{1,3}\]\.?\s+[A-Z]"  # [1] Author, - [3] Author, 1. Author
    r"|"
    r"^<sup>\[\d{1,3}\]</sup>\s*[A-Z]"  # <sup>[1]</sup> Author
)
# Also detect "Received: ... / Published: ..." date blocks
RECEIVED_RE = re.compile(
    r"^\s*>?\s*Received:?\s",
    re.IGNORECASE,
)


def find_back_matter_start(lines: list[str]) -> int | None:
    """Find the first back-matter heading or reference block."""
    earliest = None

    # 1. Look for back-matter headings
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if m:
            heading_text = m.group(2)
            if is_back_matter_heading(heading_text):
                earliest = i
                break

    # 2. Also scan for non-heading back-matter markers (bold "References" etc.)
    if earliest is None:
        for i, line in enumerate(lines):
            clean = re.sub(r"[*_`]", "", line.strip())
            if clean and is_back_matter_heading(clean) and len(clean.split()) <= 4:
                if earliest is None or i < earliest:
                    earliest = i
                break

    # 3. Detect reference lists without headings: look for consecutive
    #    numbered citation entries (e.g., [1] Author, [2] Author)
    if earliest is None:
        consecutive = 0
        ref_block_start = None
        for i, line in enumerate(lines):
            if REF_ENTRY_RE.match(line) or (line.strip().startswith("<sup>[") and "]</sup>" in line):
                if consecutive == 0:
                    ref_block_start = i
                consecutive += 1
                if consecutive >= 3:
                    # Found 3+ consecutive references — this is a reference block
                    if earliest is None or ref_block_start < earliest:
                        earliest = ref_block_start
                    break
            elif line.strip():  # non-blank, non-reference line resets counter
                consecutive = 0
                ref_block_start = None

    # 4. Detect "Received/Published" date blocks before references
    if earliest is not None:
        for i in range(max(0, earliest - 10), earliest):
            if RECEIVED_RE.match(lines[i]):
                earliest = i
                break

    return earliest


# Regex to find image references in markdown: ![...](filename)
IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

# Regex to detect figure captions — must START the line (after optional bold/whitespace)
FIGURE_CAPTION_RE = re.compile(
    r"^\s*\**\s*(?:Fig(?:ure|\.)\s*\d|FIGURE\s*\d|Scheme\s*\d|Table\s*\d)",
    re.IGNORECASE,
)


def find_captioned_images(lines: list[str]) -> set[str]:
    """Find image filenames that have a figure caption directly after them.

    Only keeps images where the next non-blank line after the image starts
    with a figure caption (Fig., Figure, FIGURE, Scheme, Table).
    This ensures only actual paper figures are kept — no logos, author
    photos, or decorative images.
    """
    captioned = set()
    for i, line in enumerate(lines):
        m = IMAGE_REF_RE.search(line)
        if not m:
            continue
        filename = m.group(1)

        # Check the image line itself
        if FIGURE_CAPTION_RE.search(line):
            captioned.add(filename)
            continue

        # Look forward for the next non-blank line — must be a caption
        for j in range(i + 1, min(i + 4, len(lines))):
            if lines[j].strip():
                if FIGURE_CAPTION_RE.search(lines[j]):
                    captioned.add(filename)
                break  # stop at first non-blank line regardless
    return captioned


def remove_uncaptioned_image_lines(lines: list[str], captioned: set[str]) -> list[str]:
    """Remove markdown image reference lines for uncaptioned images."""
    filtered = []
    for line in lines:
        m = IMAGE_REF_RE.search(line)
        if m:
            filename = m.group(1)
            if filename not in captioned:
                # Also skip the line if it's ONLY an image ref (possibly with a span)
                stripped = re.sub(r"<span[^>]*></span>", "", line).strip()
                if IMAGE_REF_RE.fullmatch(stripped):
                    continue
        filtered.append(line)
    return filtered


def trim_markdown(text: str) -> tuple[str, dict]:
    """Trim front and back matter from markdown text and filter images.

    Returns (trimmed_text, stats_dict).
    """
    lines = text.split("\n")
    stats = {
        "original_lines": len(lines),
        "title_end": None,
        "abstract_start": None,
        "back_matter_start": None,
    }

    # Find title
    title_end = find_title_end(lines)
    stats["title_end"] = title_end

    # Find abstract
    abstract_start = find_abstract_start(lines, title_end)
    stats["abstract_start"] = abstract_start

    # Find back matter
    back_matter_start = find_back_matter_start(lines)
    stats["back_matter_start"] = back_matter_start

    # Build trimmed content
    # Keep: just the title heading line + abstract onwards (abstract_start..back_matter or end)
    # title_end points to the line AFTER the title heading, so the title is at title_end-1
    if title_end > 0:
        title_lines = [lines[title_end - 1]]
    else:
        title_lines = []

    if abstract_start is not None:
        end_idx = back_matter_start if back_matter_start else len(lines)
        body_lines = lines[abstract_start:end_idx]
    else:
        # No abstract found — keep everything from title_end
        end_idx = back_matter_start if back_matter_start else len(lines)
        body_lines = lines[title_end:end_idx]

    all_kept_lines = title_lines + [""] + body_lines

    # Find which images have figure captions nearby
    captioned = find_captioned_images(all_kept_lines)
    stats["captioned_images"] = len(captioned)

    # Collect all image refs in kept text
    all_refs = set()
    for line in all_kept_lines:
        m = IMAGE_REF_RE.search(line)
        if m:
            all_refs.add(m.group(1))
    stats["total_image_refs"] = len(all_refs)
    stats["removed_images"] = list(all_refs - captioned)

    # Remove uncaptioned image lines from markdown
    all_kept_lines = remove_uncaptioned_image_lines(all_kept_lines, captioned)

    trimmed = "\n".join(all_kept_lines).strip() + "\n"
    stats["trimmed_lines"] = len(trimmed.split("\n"))
    return trimmed, stats


def process_directory(input_dir: Path, dry_run: bool = False) -> None:
    """Process all markdown files in subdirectories."""
    md_files = list(input_dir.rglob("*.md"))
    if not md_files:
        print(f"No .md files found in {input_dir}")
        return

    print(f"Found {len(md_files)} markdown files")

    total_before = 0
    total_after = 0
    no_abstract = 0
    no_backmatter = 0

    total_images_removed = 0

    for md_path in sorted(md_files):
        text = md_path.read_text(encoding="utf-8", errors="replace")
        trimmed, stats = trim_markdown(text)

        total_before += stats["original_lines"]
        total_after += stats["trimmed_lines"]

        if stats["abstract_start"] is None:
            no_abstract += 1
        if stats["back_matter_start"] is None:
            no_backmatter += 1

        # Identify image files to delete:
        # 1. Images removed from markdown (uncaptioned or in trimmed sections)
        # 2. Any image files in the directory not referenced in trimmed text
        parent_dir = md_path.parent
        image_files_on_disk = set(
            f.name for f in parent_dir.iterdir()
            if f.suffix.lower() in (".jpeg", ".jpg", ".png", ".gif", ".svg", ".webp")
        )
        # Images referenced in the final trimmed markdown
        kept_images = set()
        for line in trimmed.split("\n"):
            m = IMAGE_REF_RE.search(line)
            if m:
                kept_images.add(m.group(1))

        images_to_delete = image_files_on_disk - kept_images
        total_images_removed += len(images_to_delete)

        reduction = stats["original_lines"] - stats["trimmed_lines"]
        if reduction > 0 or images_to_delete:
            pct = reduction / stats["original_lines"] * 100 if stats["original_lines"] else 0
            if dry_run:
                msg = f"  [DRY RUN] {md_path.parent.name}: {stats['original_lines']}→{stats['trimmed_lines']} lines (-{pct:.0f}%)"
                if images_to_delete:
                    msg += f", {len(images_to_delete)} images to remove"
                print(msg)
            else:
                md_path.write_text(trimmed, encoding="utf-8")
                for img_name in images_to_delete:
                    (parent_dir / img_name).unlink()

    reduction_pct = (total_before - total_after) / total_before * 100 if total_before else 0
    print(f"\n--- Summary ---")
    print(f"Files processed: {len(md_files)}")
    print(f"Total lines: {total_before} → {total_after} (-{reduction_pct:.1f}%)")
    print(f"Images removed: {total_images_removed}")
    print(f"No abstract found: {no_abstract}")
    print(f"No back matter found: {no_backmatter}")
    if dry_run:
        print("(Dry run — no files modified)")


def main():
    parser = argparse.ArgumentParser(description="Trim front/back matter from Marker markdown")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory with markdown subdirs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying files")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: {args.input_dir} not found")
        sys.exit(1)

    process_directory(args.input_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
