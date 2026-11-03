"""Build a corpus for the pipeline from markdown papers and knowledge-graph triples.

python pipeline/scripts/build_corpus.py <markdown_dir> <out_dir> [<triplets_dir>]

<markdown_dir> holds one folder per paper, at any depth, named by DOI with the first "/"
replaced by "_", with the paper as <name>.md next to its images. PDF-to-markdown converters
such as Marker write this layout. <triplets_dir> holds one JSON list per paper of
{"node_1": str, "node_2": str, "edge": str, "paper": str}.
"""

import json
import re
import shutil
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "figures").mkdir(parents=True, exist_ok=True)

with open(out / "chunks.jsonl", "w") as f:
    for md in sorted(src.glob("**/*.md")):
        doi = md.stem.replace("_", "/", 1)
        text = re.split(r"\n#+\s*[\d.\s]*\**references", md.read_text(), flags=re.I)[0]
        # A figure is an image line followed by a caption paragraph that starts with "Fig". The
        # lookahead keeps the next image line available when a figure has several images.
        # Converters sometimes reference an image they did not write, so the file must exist.
        for image, caption in re.findall(r"!\[\]\((\S+)\)(?=\s*\n\s*\n(.+))", text):
            if re.match(r"\W*fig", caption, re.I) and (md.parent / image).exists():
                name = f"{md.stem}_{image}"
                shutil.copy(md.parent / image, out / "figures" / name)
                f.write(json.dumps({"id": f"{md.stem}#{Path(image).stem}", "source": doi,
                                    "text": caption.strip(), "image": f"figures/{name}"}) + "\n")
        # Passages are whole paragraphs packed into chunks of at most about 2,000 characters,
        # which fits the context of small embedding models.
        chunks = [""]
        for para in re.sub(r"!\[\]\(\S+\)", "", text).split("\n\n"):
            if len(chunks[-1]) + len(para) > 2000 and chunks[-1].strip():
                chunks.append("")
            chunks[-1] += para + "\n\n"
        for n, chunk in enumerate(c.strip() for c in chunks if c.strip()):
            f.write(json.dumps({"id": f"{md.stem}#{n}", "source": doi, "text": chunk}) + "\n")

if len(sys.argv) > 3:
    with open(out / "kg.jsonl", "w") as f:
        for path in sorted(Path(sys.argv[3]).glob("**/*.json")):
            for t in json.loads(path.read_text()):
                f.write(json.dumps({"head": t["node_1"], "relation": t["edge"], "tail": t["node_2"],
                                    "source": t["paper"].replace("_", "/", 1)}) + "\n")
