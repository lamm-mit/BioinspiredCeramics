"""The agent graph. Each node is one agent with its own model and instructions.

planner -> retriever -> figure_reader -> designer -> architect -> geometry -> engineer <-> tools -> judge -> fabricator -> reporter

The designer can send the retriever back once, for design fields that lack evidence. A check
compares each design value with the source the designer cites, and the designer gets one repair
with its findings. A field that still fails is dropped. A design that is still incomplete goes
straight to the reporter, so no simulation runs on it. The architect turns the structure of the
biological model into a toolpath pattern from evidence on that structure, and a complete design
gets the geometry its pattern describes. The engineer then calls tools until it has every result, and each call is recorded
in one handoff file per tool. The judge then weighs the candidate against the criteria, and the
candidate gets a novelty score against the corpus. Its reward is that score when the judge
accepts it and 0 otherwise. A candidate the judge sends back, or an accepted one whose reward is
below NOVELTY_TARGET, returns the designer to work, for up to MAX_CANDIDATES candidates per run.
The fabricator writes the synthesis and fabrication recipe of the accepted candidate with the
highest reward, and code turns its geometry into the print files of the chosen process. The
prompts are generic, and the problem itself is defined by design.py and the tools package.
"""

import json
import operator
import re
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import APIError

from tools import RUN, TOOLS

from . import corpus as cp
from . import handoff
from . import novelty as nv
from .design import ASK, CRITERIA, DESIGN, LENGTH, RANGE, SCALE, UNIT, cross_checks, design_errors, make_geometry
from .fabrication import DEFAULTS, write_fabrication
from .llm import ask_json, chat, embed, user
from .pattern import PATTERN, complete, describe, pattern_errors, road_width
from .render import render_part

# Model and reasoning effort per agent. A text model does the text jobs and a vision model
# reads the figures, with thinking off.
MODELS = {"planner": ("gpt-oss:20b", "low"),
          "figure_reader": ("qwen3.6:35b", "none"),
          "designer": ("gpt-oss:20b", "medium"),
          "architect": ("gpt-oss:20b", "medium"),
          "checker": ("gpt-oss:20b", "medium"),
          "engineer": ("gpt-oss:20b", "low"),
          "judge": ("gpt-oss:20b", "medium"),
          "fabricator": ("gpt-oss:20b", "medium"),
          "reporter": ("gpt-oss:20b", "low")}
EMBEDDING = "embeddinggemma"
MAX_CANDIDATES = 3  # designs simulated and judged in one run
# An accepted candidate with a novelty below this sends the designer back for a less reported combination.
NOVELTY_TARGET = 0.5
FIELDS = json.dumps(DESIGN)
OPTIONS = sorted({option for field in DESIGN.values() for option in field.get("enum", field.get("examples", ()))})

PLANNER = """You plan a literature search for a design question. The design fields are below.
Return JSON with keys
"queries": a list with one short search query per design field, phrased the way a paper would state it,
"entities": a list of up to five materials or concepts to look up in a knowledge graph.
Reply with the JSON only."""

FIGURE_READER = """You read one figure from a scientific paper.
Report every numerical value you can read in the figure that bears on the design fields below.
A bar height or a curve point read against its axis counts.
Return JSON {"values": [{"quantity": str, "material": str, "value": number, "unit": str, "where": str}], "summary": str}.
"where" says where in the figure the value is read, such as a bar against the y-axis, an axis tick, a label or a scale bar.
Name each quantity as the figure names it. Leave out a quantity the figure does not show.
Never give a typical or assumed value, and never give 0 for a value you cannot read."""

DESIGNER = """You fill in a design from a question and evidence. Each design field has a description
of what it needs. Apply these rules in order.
1. A value the question itself states is given. Use it, and write "question" as its source.
2. A measured property must be stated in the evidence.
3. Any other choice, such as a material or a dimension, needs evidence that supports it, for
example the same or a similar choice made or tested in a paper.
Cite the evidence id for each value, such as E7. Return JSON with keys
"design": {field: value},
"sources": {field: "evidence id: one-line reason"},
"queries": a list of search queries for the fields that no evidence supports.
Leave a field out only when no evidence supports it. Reply with the JSON only."""

ARCHITECT = """You translate the structure of a biological material into a toolpath pattern for printing a ceramic part.
The pattern parameters are listed below with what each means. A parameter you leave out stays at its default, which
turns it off, and the defaults together give flat layers. Lengths are in road widths or in ceramic layer thicknesses,
so give the ratios the evidence states for the natural structure, such as tablet length over tablet thickness, and
angles in degrees. Set the parameters that reproduce the features the evidence describes for the biological material
of the design. Give each a source, the evidence id that states it, such as A3, with the numbers used, or
"assumption: " and a reason.
Return JSON with keys
"features": [{"feature": str, "evidence": str, "parameters": [str]}], one entry per feature of the natural structure
that the evidence describes, with the parameters that carry it,
"pattern": {parameter: value},
"sources": {parameter: str}.
Reply with the JSON only."""

ENGINEER = """You run the simulations for a design with the tools.
Take every tool input from the design, the geometry or an earlier tool result, as the tool
descriptions say. Use no other numbers. If an input is available from none of them, do not call
that tool and say which input is missing. When every tool has run, reply in two sentences and call no tool."""

CHECKER = """You check one value of a design against its source text. Answer from the source text only.
Return JSON {"low": number or null, "high": number or null, "unit": str} with the smallest and the largest value
the source gives for the quantity asked, in the unit asked. For a single value, low equals high. Use null for both
when the source gives no value for that quantity. Reply with the JSON only."""

JUDGE = """You judge one candidate design against the results of its simulations.
Decide each criterion below from the design values and the tool outputs alone, and quote the value
that decides it. When they do not bear on a criterion, count it as met and say that it was not tested.
Earlier candidates of the same run and their verdicts are given too. Judge a requirement that depends on
them with them, such as a question that asks to change a choice once it fails.
Return JSON with keys
"criteria": [{"criterion": str, "met": true or false, "evidence": str}], one entry per criterion,
"change": {field: what to change and why} for the design fields or pattern parameters a new candidate should
change, empty when every criterion is met. A change must stay within the options and the description of the field,
"summary": one sentence.
Reply with the JSON only."""

FABRICATOR = """You write the recipe to make one designed ceramic laminate by additive manufacturing, from the
evidence below. Cover in order the ceramic feedstock (powder, particle size, solids loading, binder, dispersant,
mixing), the interlayer material and how it is applied between the ceramic layers, the print on the chosen
process, debinding, sintering and finishing. Use the ceramic and the interlayer of the design, even where the
evidence used other materials. A polymer interlayer burns out during sintering, so it goes in after sintering, for
example by infiltrating the gaps left by a sacrificial layer. Give every number with its unit and cite the evidence id that
states it, such as F3. A number that no evidence states is allowed only as an assumption, with the source
"assumption: " and a reason.
Return JSON with keys
"steps": [{"stage": str, "instruction": str, "parameters": {name: {"value": number, "unit": str, "source": str}}}],
"print_parameters": {name: {"value": number, "unit": str, "source": str}} for the print parameter names listed below,
"cautions": [str].
Reply with the JSON only."""

REPORTER = """Write a short report in plain markdown, without emoji, for a scientist.
Start with the outcome as given. Then write the sections listed under "sections", in that order, each with its
title as a heading, and in each write what its "write" names, and no other section. Tables of the design, the
cross-checks, the novelty and the tool outputs are appended after your text, so do not repeat them. Use only numbers, materials, criteria and
steps that appear in the input, and never supply a typical or assumed value of your own. Each tool output names
in its method the model behind its numbers. Name it where you use them. A recipe
value whose source starts with assumption does not come from the evidence. Say so. Figures, tool outputs and file
lists are appended after your text, so leave them out."""


class State(TypedDict, total=False):
    question: str
    folder: str
    queries: list
    entities: list
    proposals: list
    round: int
    passages: Annotated[list, operator.add]
    triples: Annotated[list, operator.add]
    figure_notes: Annotated[list, operator.add]
    design: dict
    sources: dict
    pattern: dict
    pattern_sources: dict
    features: list
    missing: list
    rejected: dict
    candidate: str
    geometry: dict
    messages: list
    results: list
    verdict: dict
    candidates: Annotated[list, operator.add]
    best: str
    recipe: dict
    fabrication: dict
    outcome: str
    report: str
    trace: Annotated[list, operator.add]


def plain(text):
    """Text with LaTeX spacing and units as plain characters, so that $140~\\mu m$ reads 140 µm.

    An uncertainty after ± is dropped, and 100~200 becomes the range 100-200.
    """
    text = re.sub(r"(?<=\d)~(?=\d)", "-", text.replace("\\mu", "µ").replace("\\pm", "±"))
    text = re.sub(r"\\(?:sim|approx|rm|mathrm|text)\b|\\[,;]|[$~{}]", " ", text)
    text = re.sub(r"(?:±|\+/-)\s*\d+(?:\.\d+)?", "", text)
    text = re.sub(r"(?<![\d,])(\d{1,3}),(\d{3})(?![\d,])", r"\1\2", text)  # 1,040 is one number
    return re.sub(r"µ\s+m", "µm", text)


NUMBER = r"\d+(?:\.\d+)?"


def quantities(text, units):
    """The values that stand before one of the units, as (low, high) spans in the field's own unit.

    A single value is a span of no width. A range such as 0.25 to 1.27 mm is one span. Each item
    of a list such as 50, 75 and 100 µm is its own span.
    """
    unit = "|".join(sorted(map(re.escape, units), key=len, reverse=True))
    run = rf"({NUMBER}(?:(?:\s*(?:to|and|or|[-–—/,]))+\s*{NUMBER})*)\s*\)?\s*-?\s*({unit})\b"
    items = [[float(n) * units[m[2]] for n in re.findall(NUMBER, item)]
             for m in re.finditer(run, plain(text)) for item in re.split(r",|/|\band\b|\bor\b", m[1])]
    return [(min(n), max(n)) for n in items if n]


def table_values(text, units):
    """The numbers in each table row that names one of the units, as in Young's modulus, GPa | 325 | 360."""
    return [float(n) * factor for row in plain(text).splitlines() if "|" in row for unit, factor in units.items()
            if re.search(rf"(?<!\w){re.escape(unit)}(?!\w)", row) for n in re.findall(rf"(?<![\w.]){NUMBER}(?![\w.])", row)]


def evidence(s):
    """What the designer saw, as id -> text. Ids are spelled with _ for /, so both spellings of a DOI match."""
    e = {p["id"]: p["text"][:2000] for p in s["passages"] if "image" not in p}
    # Only the caption and the values read count for a figure. Its summary is the reader's own prose.
    e |= {f["id"]: f["caption"] + " " + " ".join(f"{v.get('value')} {v.get('unit')}" for v in f["reading"].get("values")
                                                  or [] if isinstance(v, dict)) for f in s["figure_notes"]}
    for t in s["triples"]:
        doi = t["source"].replace("/", "_")
        e[doi] = f"{e.get(doi, '')} {t['head']} {t['relation']} {t['tail']}."
    return e | {"question": s["question"]}


def reading(v):
    """True for a figure value that is a number with a place in the figure. The reader sometimes admits a value is not shown."""
    return (isinstance(v, dict) and isinstance(v.get("value"), (int, float)) and bool(v.get("where"))
            and not re.search(r"\bnot (?:printed|shown|given|visible|stated|present)|assum|typical|standard", str(v["where"]), re.I))


def shown(s):
    """The evidence lines the designer reads, as (id, text) in a fixed order.

    The designer cites a line by its position, as E1, E2 and so on. Models copy a short label
    reliably, and they miscopy a long id such as a DOI with a passage number.
    """
    rows = [(p["id"], p["text"][:2000]) for p in s["passages"] if "image" not in p]
    rows += [(f["id"], f"figure with caption: {f['caption'][:500]} Reading: {json.dumps(f['reading'])}")
             for f in s["figure_notes"]]
    return rows + [(t["source"], arrow(t)) for t in s["triples"]]


def arrow(t):
    """A knowledge graph triple as an evidence line."""
    return f"{t['head']} -> {t['relation']} -> {t['tail']}"


def cited(source, e):
    """The evidence ids a source names. The word question counts only at the start, not inside a reason."""
    source = str(source).replace("/", "_")
    ids = [i for i in e if i != "question" and re.search(re.escape(i) + r"(?![\w#])", source)]
    return ids + ["question"] * bool(re.match(r"\W*question\b", source, re.I))


def source_errors(design, sources, s):
    """The fields whose source fails the check, as field -> reason.

    A source must name an id among the evidence. A numeric value must equal a value in that
    evidence or lie within a range it states. A field in RANGE may lie anywhere between the
    smallest and the largest value there.
    """
    e, errors = evidence(s), {}
    for field, value in design.items():
        ids = cited(sources.get(field, ""), e)
        # A paper id alone covers its graph triples. A value from a passage needs the passage id.
        hint = "" if any("#" in i for i in ids) else " Cite the id of the passage or figure that states it."
        if not ids:
            errors[field] = "the source names no id from the evidence." + hint
        elif field in SCALE and not isinstance(value, (int, float)):
            errors[field] = "the value is not a number"
        elif field in SCALE:
            spans = [x for i in ids for x in quantities(e[i], SCALE[field])]
            stated = spans + [(x, x) for i in ids for x in table_values(e[i], SCALE[field])]
            slack = 0.005 * abs(value)
            holds = lambda spans: any(low - slack <= value <= high + slack for low, high in spans)
            if not (holds(stated) or field in RANGE and spans and min(spans)[0] <= value <= max(high for _, high in spans)):
                unit = next(u for u, factor in SCALE[field].items() if factor == 1)
                values = ", ".join(f"{low:g}" if low == high else f"{low:g} to {high:g}" for low, high in sorted(set(stated))[:12])
                found = f"gives {values} {unit}, not {value}." if stated else f"gives no value in {unit}."
                # Among 100 lines a model misremembers which one holds a number, and cites the wrong one again on
                # repair. So the lines that state the value are named.
                others = [i for i in e if i not in ids and holds(quantities(e[i], SCALE[field])
                                                                 + [(x, x) for x in table_values(e[i], SCALE[field])])]
                found += f" {value:g} {unit} is stated by {' and '.join(others[:3])}." if others else ""
                errors[field] = f"{' and '.join(ids)} {found}{hint}"
    return errors


def accepted(s):
    return [c for c in s.get("candidates", []) if c["verdict"]["verdict"] == "accept"]


def after_judge(s):
    """Back to the designer while candidates remain and the last was sent back, or every accepted one lacks novelty."""
    verdict = s["verdict"]["verdict"]
    more = len(s["candidates"]) < MAX_CANDIDATES and (
        verdict == "revise" or verdict == "accept" and max(c["reward"] for c in accepted(s)) < NOVELTY_TARGET)
    return "designer" if more else "fabricator" if accepted(s) else "reporter"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(x).replace("|", "/") for x in row) + " |" for row in rows])


def recipe_markdown(recipe):
    """The recipe as steps with their parameters, each value with its unit and source."""
    lines = [f"# Recipe for {recipe['candidate']}, {recipe['process']}", ""]
    for n, step in enumerate(recipe["steps"], 1):
        lines += [f"{n}. {step.get('stage', '').capitalize()}. {step.get('instruction', '')}"]
        lines += [f"   - {k}: {p.get('value')} {p.get('unit', '')} ({p.get('source')})" for k, p in step["parameters"].items()]
    lines += ["", "Print parameters", ""] + [f"- {k}: {p.get('value')} {p.get('unit', '')} ({p.get('source')})"
                                             for k, p in recipe["print_parameters"].items()]
    return "\n".join(lines + ["", "Cautions", ""] + [f"- {c}" for c in recipe["cautions"]]) + "\n"


def appendix(s):
    """The fixed sections after the narrative, written from the state. Paths are relative to the run folder."""
    run = Path(s["folder"])
    rel = lambda p: Path(p).relative_to(run).as_posix()
    judged, out = s.get("candidates", []), []
    if judged:
        named = [k for k in DESIGN if "enum" in DESIGN[k] or "examples" in DESIGN[k]]
        out += ["## Candidates", "", table(
            ["Candidate", *named, "Structure", "Verdict", "Novelty", "Reward"],
            [[c["candidate"], *[c["design"].get(k, "") for k in named], describe(c["pattern"]), c["verdict"]["verdict"],
              f"{c['novelty']['score']:.2f}", f"{c['reward']:.2f}"] for c in judged]), ""]
    if s.get("proposals"):
        out += ["## Knowledge graph proposals", "", "Hypotheses whose every link a triple of the knowledge graph names, while no "
                "passage, caption or triple names all their terms. The number after a link counts its triples.", "", table(
                    ["Hypothesis", "Structure", "Links", "Taken by"],
                    [[", ".join(p["design"].values()), p["structure"],
                      "; ".join(f"{link['terms'][0]} with {link['terms'][1]}: {link['triples']}" for link in p["links"]),
                      ", ".join(c["candidate"] for c in judged if all(c["design"].get(k) == v for k, v in p["design"].items()))]
                     for p in s["proposals"]]), ""]
    if s.get("best"):
        best = next(c for c in judged if c["candidate"] == s["best"])
        out += [f"## Design of {best['candidate']}", "", table(["Field", "Value", "Source"],
                [[k, v, best["sources"].get(k, "")] for k, v in best["design"].items()]), ""]
        out += [f"## Structure of {best['candidate']}", "", describe(best["pattern"]) + ".", "",
                table(["Parameter", "Value", "Unit", "Source"], [[k, x["value"], PATTERN[k].get("unit", ""), x["label"]]
                                                                 for k, x in best["pattern_sources"].items()]), ""]
        out += [f"- {f.get('feature')} ({f.get('evidence')}): {', '.join(map(str, f.get('parameters') or []))}"
                for f in best["features"] if isinstance(f, dict)] + [""] * bool(best["features"])
        fem = next((r["output"] for r in best["results"] if r["output"].get("damage_png")), {})
        figures = [("Structure of the part", s["fabrication"]["files"].get("render_png")),
                   ("Damage after the crack simulation", fem.get("damage_png")),
                   ("Print plan", s["fabrication"]["files"].get("preview_png"))]
        out += ["## Figures", ""] + [f"![{name}]({rel(path)})" for name, path in figures if path] + [""]
        out += ["## Novelty of the best candidate", "", best["novelty"]["statement"], "",
                table(["Component", "Value"], [[k, best["novelty"][k]] for k in ["score", "combination", "semantic",
                      "max_terms_together", "records_with_max", "closest_records", "nearest_passage", "nearest_similarity"]]), ""]
    if judged:
        out += ["## Cross-checks", "", table(["Candidate", "Check", "Design value", "Check value", "Relative difference", "Method"],
                [[c["candidate"], name, x.get("design_layer_modulus_GPa", ""), x.get("youngs_modulus_GPa", x.get("error", "")),
                  x.get("relative_difference", ""), x.get("method", "")] for c in judged for name, x in c["cross_checks"].items()]), ""]
    if s.get("missing"):
        out += ["## Incomplete design", "", "The last design proposed lacked these fields, so it did not reach the tools"
                + (". It is not the best candidate." if s.get("best") else "."), "",
                table(["Field", "Reason"], [[k, s.get("rejected", {}).get(k, "no value proposed")] for k in s["missing"]]), ""]
    for c in judged:
        out += [f"## Simulation outputs of {c['candidate']}", "", table(["Tool", "Inputs", "Outputs", "Method"], [
            [r["tool"], ", ".join(f"{k} = {v}" for k, v in r["arguments"].items() if not str(v).endswith(".json")),
             ", ".join(f"{k} = {v}" for k, v in r["output"].items() if k != "method" and not str(v).endswith((".png", ".csv", ".json"))),
             r["output"].get("method", "")] for r in c["results"]]
            + [[name, c["design"].get("ceramic", ""), ", ".join(f"{k} = {v}" for k, v in out_.items() if k != "method"),
                out_.get("method", "cross-check")] for name, out_ in c["cross_checks"].items()]), ""]
    if s.get("recipe"):
        out += ["## Recipe", "", recipe_markdown(s["recipe"]).split("\n", 2)[2], "## Print files", "",
                table(["File", "Path"], [[k, rel(v)] for k, v in s["fabrication"]["files"].items()]), ""]
    return "\n".join(out)


def build(models, corpus):
    def ask(agent, system, text, images=()):
        model, effort = models[agent]
        try:
            out, rec = ask_json(model, system, text, images, effort)
        except ValueError as e:
            # A thinking model can spend its whole token budget on reasoning and return no answer, and a reply
            # can hold broken JSON, such as a list left open. Either gets one more try at low effort, with the
            # error added, since at temperature 0 the same prompt gives the same reply.
            again = f"{text}\n\nYour last reply could not be read: {str(e)[:300]}\nReply with one valid JSON object only."
            out, rec = ask_json(model, system, again, images, effort if effort in (None, "none", "low") else "low")
            rec["retried"] = True
        return out, rec | {"agent": agent}

    def call(agent, messages, tools=None):
        model, effort = models[agent]
        reply, rec = chat(model, messages, tools, effort)
        return reply, rec | {"agent": agent}

    def planner(s):
        # Hypotheses the knowledge graph supports link by link and no record names whole, found in code. Their link
        # triples join the evidence, so the designer can cite them.
        found = nv.proposals(corpus, s["question"])
        text = f"Question: {s['question']}\nDesign fields: {FIELDS}"
        if found:
            text += ("\nThe designer may choose one of these hypotheses, which no passage or triple names whole. Include queries "
                     f"that would find the layer thicknesses and moduli of their materials: "
                     f"{json.dumps([p['design'] | {'structure': p['structure']} for p in found])}")
        out, rec = ask("planner", PLANNER, text)
        return {"queries": out["queries"], "entities": out["entities"], "round": 0, "proposals": found,
                "triples": [link["example"] for p in found for link in p["links"]], "trace": [rec]}

    def retriever(s):
        seen = {p["id"] for p in s["passages"]}
        found = [p for q in s["queries"]
                 for p in cp.search(corpus, q, 4, skip=seen) + cp.search(corpus, q, 1, figures=True, skip=seen)]
        # Semantic search favors passages on the topic over passages that state a value or name a
        # material. So an open field is also searched among the passages that can supply it. Those
        # hold a number with the field's unit, or the name of one of its options. The first round
        # does this for the numeric fields, and the second for every field still missing.
        # The second round searches with the choices the design already holds, such as alumina and direct ink writing.
        chosen = " ".join(v for v in s.get("design", {}).values() if isinstance(v, str))
        names = {"ceramic": "ceramic", "interlayer": "interlayer"} | s.get("design", {})
        for field in s.get("missing", UNIT):
            query = f"{s['question']} {chosen} {DESIGN[field]['description']}"
            if field in UNIT:
                found += cp.search(corpus, query, 4, skip=seen, pattern=UNIT[field])
            if field in ASK and s.get("design"):
                # The topic of the question, such as nacre, favors passages on composites, so the second round
                # also asks for the quantity of the design's own material alone, as the checker reads it.
                found += cp.search(corpus, ASK[field].split(".")[0].format(**names), 4, skip=seen, pattern=UNIT[field])
            for option in DESIGN[field].get("enum", DESIGN[field].get("examples", ())):
                # Papers name a process or a material in several ways, such as DLP for vat photopolymerization.
                names = "|".join(re.escape(n) for n in [option, *nv.ALIASES.get(option, ())])
                found += cp.search(corpus, query, 1, skip=seen, pattern=f"(?i)(?:{names})")
        # The knowledge graph is asked about the planner's entities and about every allowed design option.
        triples = cp.neighbors(corpus, s["entities"] + OPTIONS) if s["round"] == 0 else []
        return {"passages": list({p["id"]: p for p in found}.values()), "triples": triples, "round": s["round"] + 1}

    def figure_reader(s):
        done = {f["id"] for f in s["figure_notes"]}
        notes, trace = [], []
        for p in s["passages"]:
            if "image" in p and p["id"] not in done:
                try:
                    out, rec = ask("figure_reader", FIGURE_READER, f"Caption: {p['text']}\nDesign fields: {FIELDS}",
                                   [corpus["folder"] / p["image"]])
                except (APIError, ValueError) as e:  # the server can fail on one image, as with a GPU fault
                    trace.append({"agent": "figure_reader", "figure": p["id"], "error": f"{type(e).__name__}: {str(e)[:200]}"})
                    continue
                out["values"] = [v for v in out.get("values") or [] if reading(v)]
                notes.append({"id": p["id"], "caption": p["text"], "reading": out})
                trace.append(rec | {"figure": p["id"]})
        return {"figure_notes": notes, "trace": trace}

    def designer(s):
        rows = shown(s)
        judged = s.get("candidates", [])
        # After a second search the fields that passed stay, and only the missing ones are asked for. After the
        # judge sent a candidate back, the design starts over.
        fresh = not s.get("design") or bool(judged) and s["design"] == judged[-1]["design"]
        kept = {} if fresh else s["design"]
        # A number still missing after the first search frees the material it belongs to, with every number kept for
        # that material. The corpus may state no number for a material, such as one a knowledge graph proposal holds,
        # and the designer can then choose another. The proposals are not offered again.
        names = lambda f: re.findall(r"\{(\w+)\}", ASK[f])  # the materials a checker question names, its own first
        freed = {names(f)[0] for f in s.get("missing", []) if f in ASK} if kept else set()
        kept = {k: v for k, v in kept.items() if k not in freed and not (k in ASK and freed & set(names(k)))}
        prompt = (f"Question: {s['question']}\nDesign fields: {FIELDS}\nEvidence:\n"
                  + "\n".join(f"[E{n}] {text}" for n, (_, text) in enumerate(rows, 1)))
        # Models quote a typical value from memory and cite a line that does not state it. So the numbers each
        # line states in the unit of each numeric field are listed, found the way the source check finds them.
        e = evidence(s)
        stated = {field: [f"E{n}: " + ", ".join(f"{lo:g}" if lo == hi else f"{lo:g} to {hi:g}" for lo, hi in spans[:8])
                          for n, (i, _) in enumerate(rows, 1)
                          for text in [e.get(i.replace("/", "_"), e.get(i, ""))]
                          if re.search(UNIT[field], plain(text)) for spans in [quantities(text, SCALE[field])] if spans][:20]
                  for field in SCALE}
        prompt += ("\nNumbers that the evidence lines state in the unit of each numeric field. A numeric value must be one "
                   "of the numbers of the line it cites:\n" + json.dumps(stated))
        if s.get("proposals") and not freed:
            line = {text: f"E{n}" for n, (_, text) in enumerate(rows, 1)}
            links = lambda p: ", ".join(f"{link['terms'][0]} with {link['terms'][1]} ({line[arrow(link['example'])]})" for link in p["links"])
            prompt += ("\nHypotheses that the knowledge graph proposes, each with a structure that the architect builds after "
                       "you. A triple of the graph names each link, given with its evidence line, and no passage or triple names "
                       "all the terms together. You may choose one, citing those lines for its choices, when the evidence also "
                       "supplies its numeric fields:\n"
                       + "\n".join(f"- {json.dumps(p['design'])} with {p['structure']}, links {links(p)}" for p in s["proposals"]))
        if judged:
            prompt += ("\nEarlier candidates of this run, with the judge's verdict and the novelty of each against the "
                       f"corpus, from 0 for a combination already reported together to 1 for one never reported "
                       f"(target {NOVELTY_TARGET}):\n"
                       + json.dumps([{"design": c["design"], "judge": c["verdict"], "novelty": c["novelty"]["score"]}
                                     for c in judged])
                       + "\nPropose a different design, with evidence for every value as before. Where the judge asks to "
                       "change a field, change it even when the question states its value. Where the judge accepted a "
                       "candidate whose novelty is below the target, choose a combination of biological model, mechanism, "
                       "materials and process that the corpus has not reported together, such as a knowledge graph proposal.")

        def real(text):  # E7 back to the id it stands for
            return re.sub(r"\bE(\d+)\b", lambda m: rows[int(m[1]) - 1][0] if 0 < int(m[1]) <= len(rows) else m[0], text)

        def labels(text):  # and an id back to its label, longest id first since a DOI starts its passage ids
            for n, (i, _) in sorted(enumerate(rows, 1), key=lambda row: -len(row[1][0])):
                text = text.replace(i.replace("/", "_"), f"E{n}")
            return text

        def second_reading(design):
            """Numeric fields whose value a second reading of the cited source does not give, as field -> reason.

            The source check finds a value in its source. It cannot tell which of several values there
            belongs to the field, such as the thickness of another layer in the same sentence. The
            checker is asked a concrete question about the source alone, with the materials of the design named.
            """
            e, errors = evidence(s), {}
            for field in ASK.keys() & design.keys() - kept.keys():  # a kept field was read in an earlier call
                try:
                    quantity = ASK[field].format(**design)
                except KeyError:  # the question names a field the design does not have yet
                    continue
                ids = cited(sources[field], e)
                text = "\n".join(f"[{i}] {e[i]}" for i in ids)
                unit = next(u for u, factor in SCALE[field].items() if factor == 1)
                if (field, quantity, text) not in read:  # a repair that keeps a source needs no new reading
                    out, rec = ask("checker", CHECKER, f"Quantity: {quantity}\nUnit: {unit}\nSource:\n{text}")
                    trace.append(rec)
                    read[field, quantity, text] = out.get("low"), out.get("high")
                low, high = read[field, quantity, text]
                slack = 0.005 * abs(design[field])
                if not (isinstance(low, (int, float)) and isinstance(high, (int, float))
                        and low - slack <= design[field] <= high + slack):
                    found = "no value" if not isinstance(low, (int, float)) else f"{low:g} {unit}" if low == high else f"{low:g} to {high:g} {unit}"
                    errors[field] = f"a second reading of {' and '.join(ids)} finds {found} for the {quantity} Your value is {design[field]}."
            return errors

        def repeats(design):
            """The fields to change when a design repeats a judged candidate, as field -> reason."""
            if not any(design == c["design"] for c in judged) or all(k in PATTERN for k in judged[-1]["verdict"].get("change") or [0]):
                return {}  # a change of the pattern alone keeps the design
            named = [k for k in judged[-1]["verdict"].get("change", {}) if k in design] or list(design)
            return {k: "the design repeats a candidate that the judge sent back. Change this field." for k in named}

        if kept:
            prompt += (f"\nThese fields passed the check of their sources and stay as they are: {json.dumps(kept)}\n"
                       f"Fill only the missing fields: {json.dumps([k for k in DESIGN if k not in kept])}")
        design, sources, rejected, trace, read = dict(kept), {k: s["sources"][k] for k in kept}, {}, [], {}
        for attempt in range(2):  # the second attempt is the one repair, with the findings of the source check
            out, rec = ask("designer", DESIGNER, prompt)
            trace.append(rec)
            # A repair may answer only the rejected fields, so each answer adds to the fields that passed.
            design |= {k: v for k, v in (out.get("design") or {}).items() if k in DESIGN and k not in kept and v is not None}
            sources |= {k: real(str(v)) for k, v in (out.get("sources") or {}).items() if k not in kept}
            failed = source_errors(design, sources, s) | design_errors(design) | repeats(design)
            failed |= second_reading({k: v for k, v in design.items() if k not in failed})
            failed = {k: v for k, v in failed.items() if k not in kept}
            design = {k: v for k, v in design.items() if k not in failed}
            rejected |= failed
            if not failed:
                break
            prompt += (f"\nYour last answer was {json.dumps(out)}\nA check of each value against its source "
                       f"rejected these fields: {labels(json.dumps(failed))}\nAnswer again. Give each rejected field "
                       "a value and a source that pass, or leave it out.")
        return {"design": design, "sources": sources, "missing": [k for k in DESIGN if k not in design],
                "rejected": {k: v for k, v in rejected.items() if k not in design},
                "queries": out.get("queries", []), "trace": trace}

    def architect(s):
        """The toolpath pattern of the design, from evidence on the structure of its biological model.

        Each parameter keeps its source with a label from code. It is cited when the line states the value,
        derived when the line states two lengths whose ratio is the value, and an assumption otherwise. A
        pattern that fails its checks gets one repair, then loses the failing parameters.
        """
        d, judged = s["design"], s.get("candidates", [])
        bio = d["biological_model"]
        queries = [f"{bio} microstructure, tablet, lamella or fiber dimensions and aspect ratio",
                   f"{bio} orientation of layers, pitch angle, rotation, crossed or concentric arrangement",
                   f"{bio} waviness, interlocking or suture of the interfaces",
                   f"{bio} gradient of modulus or composition through the thickness",
                   f"{bio}-inspired ceramic made by {d['process']}, toolpath, raster or filament arrangement"]
        rows = list({p["id"]: p for q in queries for p in cp.search(corpus, q, 3)}.values())
        prompt = (f"Design: {json.dumps(d)}\nRoad width: {road_width(d) * 1000:.0f} um. Ceramic layer thickness: "
                  f"{d['layer_thickness_um']:g} um.\nPattern parameters: {json.dumps(PATTERN)}\nEvidence:\n"
                  + "\n".join(f"[A{n}] {p['text'][:1500]}" for n, p in enumerate(rows, 1)))
        taken = next((p for p in s.get("proposals", []) if all(d.get(k) == v for k, v in p["design"].items())), None)
        if taken:
            prompt += (f"\nThe design follows a knowledge graph proposal whose structure is {taken['structure']}, carried by "
                       f"{nv.CARRIES[taken['structure']]}. The graph ties it to {bio} and to {d['mechanism']}: "
                       + "; ".join(arrow(link["example"]) for link in taken["links"] if taken["structure"] in link["terms"])
                       + f". Build it unless the evidence on {bio} contradicts it.")
        if judged:
            prompt += ("\nEarlier candidates of this run with their patterns and the judge's verdict:\n"
                       + json.dumps([{"design": c["design"], "pattern": c["pattern"], "judge": c["verdict"]} for c in judged])
                       + "\nWhere the judge asks to change a pattern parameter, change it.")

        def labeled(pattern, sources):
            out = {}
            for k, v in pattern.items():
                source = str(sources.get(k, ""))
                ids = [int(m) for m in re.findall(r"\bA(\d+)\b", source) if 0 < int(m) <= len(rows)]
                named = ", ".join(rows[i - 1]["id"] for i in ids)
                texts = [plain(rows[i - 1]["text"]) for i in ids]
                if not ids or source.lower().startswith("assumption"):
                    label = f"assumption: {re.sub(r'(?i)^assumption\W*', '', source) or 'no source'}"
                elif isinstance(v, bool) or not isinstance(v, (int, float)):
                    label = f"supported: {named}"
                elif any(re.search(rf"(?<![\d.]){re.escape(f'{v:g}')}(?!\d)", t) for t in texts):
                    label = f"cited: {named}"
                else:
                    lengths = [lo for t in texts for lo, hi in quantities(t, LENGTH) if lo == hi and lo > 0]
                    pair = next(((a, b) for a in lengths for b in lengths if v and abs(a / b / v - 1) <= 0.05), None)
                    # The wording says the value is absent, since "states no 10" was read as evidence against 10.
                    label = f"derived: {named}, {pair[0]:g} / {pair[1]:g}" if pair else f"assumption: {v:g} does not appear in {named}"
                out[k] = {"value": v, "source": source, "label": label}
            return out

        def repeats(pattern):
            same = [c["candidate"] for c in judged if c["design"] == d and complete(c["pattern"]) == complete(pattern)]
            return {"pattern": f"the design and pattern repeat {same[0]}, which the judge sent back. Change the pattern."} if same else {}

        trace = []
        for attempt in range(2):
            out, rec = ask("architect", ARCHITECT, prompt)
            trace.append(rec)
            pattern = {k: v for k, v in (out.get("pattern") or {}).items() if k in PATTERN and v is not None and v != PATTERN[k]["default"]}
            errors = pattern_errors(pattern, d) | repeats(pattern)
            if not errors:
                break
            prompt += (f"\nYour last answer was {json.dumps(out)}\nThe pattern failed these checks: {json.dumps(errors)}\n"
                       "Answer again with a pattern that passes.")
        dropped = {}
        while errors and pattern:  # a parameter that still fails goes back to its default, and failing defaults reset all
            failing = {k: r for k, r in errors.items() if k in pattern} or {"pattern": " ".join(errors.values())}
            dropped |= failing
            pattern = {} if "pattern" in failing else {k: v for k, v in pattern.items() if k not in failing}
            errors = pattern_errors(pattern, d)
        sources = out.get("sources") or {}
        return {"pattern": pattern, "pattern_sources": labeled(pattern, sources), "trace": trace,
                "features": [f for f in out.get("features") or [] if isinstance(f, dict)]
                            + [{"feature": f"dropped {k}", "evidence": "pattern check", "parameters": [k], "reason": r} for k, r in dropped.items()]}

    def geometry(s):
        """Start a candidate: its folder, the design and pattern with their sources, and the geometry of the part."""
        folder = Path(s["folder"]) / f"candidate_{len(s.get('candidates', [])) + 1}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "design.json").write_text(json.dumps({k: s.get(k) for k in ["design", "sources", "pattern", "pattern_sources", "features"]},
                                                       indent=1, ensure_ascii=False))
        return {"candidate": str(folder), "geometry": make_geometry(s["design"], s.get("pattern", {}), folder / "geometry"),
                "messages": [], "results": []}

    def engineer(s):
        given = user(json.dumps({"design": s["design"], "geometry": s["geometry"]}))
        reply, rec = call("engineer", [{"role": "system", "content": ENGINEER}, given] + s["messages"], TOOLS)
        return {"messages": s["messages"] + [reply], "trace": [rec]}

    def tools(s):
        messages, results = [], []
        for c in s["messages"][-1]["tool_calls"]:
            name, args, inputs = c["function"]["name"], c["function"]["arguments"], None
            try:
                args = json.loads(args)
                run = RUN[name]
                inputs = handoff.record(s | {"results": s["results"] + results}, name, args)
                out = run(**args)
            except Exception as e:  # a bad name or argument goes back to the model, which can retry
                out = {"error": f"{type(e).__name__}: {e}"}
            if inputs is not None:
                handoff.finish(s, name, out)
            messages.append({"role": "tool", "tool_call_id": c["id"], "name": name, "content": json.dumps(out)})
            results.append({"tool": name, "arguments": args, "inputs": inputs, "output": out})
        return {"messages": s["messages"] + messages, "results": s["results"] + results}

    def judge(s):
        """The verdict on the current candidate. The model decides each criterion, and the verdict follows from them here."""
        last = {r["tool"]: r for r in s["results"] if "error" not in r["output"]}
        absent = [t["function"]["name"] for t in TOOLS if t["function"]["name"] not in last]
        unmatched = [f"{r['tool']}.{k}" for r in last.values() for k, i in r["inputs"].items() if i["from"] == "unmatched"]
        trace = []
        if absent or unmatched:
            # Without every result, or with an input that came from nowhere, there is nothing sound to judge.
            verdict = {"verdict": "incomplete", "summary": " ".join(
                [f"No result from {', '.join(absent)}."] * bool(absent)
                + [f"Inputs from neither the design, the geometry nor a tool: {', '.join(unmatched)}."] * bool(unmatched))}
        else:
            facts = {"question": s["question"], "design": s["design"], "sources": s["sources"], "geometry": s["geometry"],
                     "structure": describe(s["pattern"]), "pattern": s["pattern_sources"], "features": s["features"],
                     "results": [{k: r[k] for k in ["tool", "arguments", "output"]} for r in last.values()],
                     "earlier_candidates": [{"candidate": c["candidate"], "design": c["design"], "verdict": c["verdict"]}
                                            for c in s.get("candidates", [])]}
            out, rec = ask("judge", JUDGE, f"Criteria: {json.dumps(CRITERIA)}\nDesign fields: {FIELDS}\n"
                                           f"Pattern parameters: {json.dumps(PATTERN)}\n{json.dumps(facts)}")
            trace = [rec]
            criteria = [c for c in out.get("criteria") or [] if isinstance(c, dict)]
            accept = bool(criteria) and all(c.get("met") is True for c in criteria)
            verdict = {"verdict": "accept" if accept else "revise", "criteria": criteria,
                       "change": {} if accept else {k: v for k, v in (out.get("change") or {}).items() if k in DESIGN or k in PATTERN},
                       "summary": out.get("summary", "")}
        novelty = nv.novelty(s["design"], s["pattern"], corpus, lambda texts: embed(corpus["embed_model"], texts))
        reward = novelty["score"] if verdict["verdict"] == "accept" else 0.0
        folder = Path(s["candidate"])
        (folder / "verdict.json").write_text(json.dumps(verdict | {"reward": reward}, indent=1, ensure_ascii=False))
        (folder / "novelty.json").write_text(json.dumps(novelty, indent=1, ensure_ascii=False))
        checks = cross_checks(s["design"])
        (folder / "cross_checks.json").write_text(json.dumps(checks, indent=1, ensure_ascii=False))
        candidate = {"candidate": folder.name, "design": s["design"], "sources": s["sources"], "pattern": s["pattern"],
                     "pattern_sources": s["pattern_sources"], "features": s["features"], "geometry": s["geometry"],
                     "results": s["results"], "verdict": verdict, "novelty": novelty, "reward": reward, "cross_checks": checks}
        # A candidate that is sent back may take one more search for its replacement.
        return {"verdict": verdict, "candidates": [candidate], "round": 1, "trace": trace}

    def fabricator(s):
        """The recipe and the print files of the accepted candidate with the highest reward."""
        best = max(accepted(s), key=lambda c: c["reward"])
        d, folder = best["design"], Path(s["folder"]) / best["candidate"]
        queries = [f"{d['process']} of {d['ceramic']}, slurry or ink, solids loading, binder and dispersant",
                   f"{d['process']} print parameters for {d['ceramic']}, nozzle, speed, layer thickness or exposure",
                   f"debinding and sintering of {d['ceramic']} parts made by {d['process']}",
                   f"{d['interlayer']} interlayer applied between {d['ceramic']} layers"]
        shell = complete(best["pattern"])["interlayer_placement"] == "shell"
        route = ("core-shell filaments from one coaxial head, a ceramic core inside an interlayer shell" if shell
                 else "ceramic and interlayer printed from separate tools, the interlayer as sheets between plies")
        queries.append(f"{d['process']} of {d['ceramic']} by coextrusion of core-shell filaments" if shell
                       else f"{d['process']} of {d['ceramic']} with a second material between layers")
        rows = list({p["id"]: p for q in queries for p in cp.search(corpus, q, 4)}.values())
        out, rec = ask("fabricator", FABRICATOR, f"Design: {json.dumps(d)}\nStructure: {describe(best['pattern'])}. "
                       f"Print route: {route}.\nPrint parameter names: "
                       f"{json.dumps(DEFAULTS[d['process']])}\nEvidence:\n"
                       + "\n".join(f"[F{n}] {p['text'][:2000]}" for n, p in enumerate(rows, 1)))

        def checked(p):
            """The parameter with its source as a passage id, kept only when that passage states the number."""
            m, value = re.match(r"\W*F(\d+)\b", str(p.get("source", ""))), p.get("value")
            if m and 0 < int(m[1]) <= len(rows):
                passage = rows[int(m[1]) - 1]
                text = plain(passage["text"])
                if (isinstance(value, (int, float)) and re.search(rf"(?<![\d.]){re.escape(f'{value:g}')}(?!\d)", text)
                        or isinstance(value, str) and value.strip() and value.strip().lower() in text.lower()):
                    return p | {"source": passage["id"]}
                return p | {"source": f"assumption: {passage['id']} does not state {value}"}
            source = str(p.get("source") or "no source")
            return p | {"source": source if source.startswith("assumption") else f"assumption: {source}"}

        params = lambda x: {k: checked(v) for k, v in (x or {}).items() if isinstance(v, dict)}
        recipe = {"candidate": best["candidate"], "process": d["process"],
                  "steps": [st | {"parameters": params(st.get("parameters"))} for st in out.get("steps") or [] if isinstance(st, dict)],
                  "print_parameters": params(out.get("print_parameters")), "cautions": out.get("cautions") or [],
                  "evidence": [p["id"] for p in rows]}
        (folder / "recipe.json").write_text(json.dumps(recipe, indent=1, ensure_ascii=False))
        (folder / "recipe.md").write_text(recipe_markdown(recipe))
        geometry = best["geometry"]["geometry_json"]
        made = write_fabrication(geometry, d["process"], recipe["print_parameters"], folder / "fabrication")
        made["files"]["render_png"] = str(render_part(geometry, folder / "geometry" / "render.png"))
        return {"best": best["candidate"], "recipe": recipe, "fabrication": made, "trace": [rec]}

    def reporter(s):
        judged, last = s.get("candidates", []), s.get("verdict", {})
        if accepted(s):
            best = next(c for c in judged if c["candidate"] == s["best"])
            outcome = (f"{best['candidate']} was accepted with a novelty of {best['reward']:.2f} against the corpus"
                       + f", below the target of {NOVELTY_TARGET}" * (best["reward"] < NOVELTY_TARGET))
        elif last.get("verdict") == "incomplete":
            outcome = f"the simulation of {judged[-1]['candidate']} did not complete. {last['summary']}"
        else:
            outcome = (f"no candidate was accepted after {len(judged)} of at most {MAX_CANDIDATES}" if judged else "no design was simulated")
            outcome += ", and the last design is incomplete" * bool(s.get("missing"))
        facts = {"outcome": outcome} | {k: s[k] for k in ["question", "design", "sources", "missing", "rejected"] if k in s}
        # Every judged candidate keeps its tool outputs and novelty, so the report sees each method string.
        facts["candidates"] = [{k: c[k] for k in ["candidate", "design", "sources", "verdict", "reward", "cross_checks", "features"]}
                               | {"structure": describe(c["pattern"]), "pattern": c["pattern_sources"]}
                               | {"novelty": {k: c["novelty"][k] for k in ["score", "statement", "closest_records", "nearest_passage"]},
                                  "results": [{"tool": r["tool"], "arguments": r["arguments"], "output": r["output"]} for r in c["results"]]}
                               for c in judged]
        if s.get("recipe"):
            facts |= {"best": s["best"], "recipe": s["recipe"],
                      "print_files": {k: v for k, v in s["fabrication"].items() if k != "layers"}}
        # The sections follow from what the run produced, so the report cannot describe a step that never ran.
        sections = [("Question", f"the question as asked, {s['question']!r}")]
        if s.get("recipe"):
            sections += [("Hypothesis", "the biological model of the best candidate, its toughening mechanism, the features of "
                                        "its natural structure, and how the toolpath pattern, the materials and the print "
                                        "process carry them into the ceramic")]
        if judged:
            sections += [("Simulation results", "what the tool outputs of each candidate mean, and the model behind each"),
                         ("Judge's findings", "the judge's finding on each criterion for each candidate")]
        sections += [("What remains unverified", "the models and limits the tool methods state, the pattern parameters and recipe values "
                                                 "whose source starts with assumption, the limits the crack simulation states, each cross-check whose relative difference exceeds 0.2 in size, "
                                                 "and that no part has been made or tested")]
        facts["sections"] = [{"title": t, "write": w} for t, w in sections]
        reply, rec = call("reporter", [{"role": "system", "content": REPORTER}, user(json.dumps(facts))])
        return {"report": reply["content"].rstrip() + "\n\n" + appendix(s), "outcome": outcome, "trace": [rec]}

    g = StateGraph(State)
    for node in [planner, retriever, figure_reader, designer, architect, geometry, engineer, tools, judge, fabricator, reporter]:
        g.add_node(node.__name__, node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "figure_reader")
    g.add_edge("figure_reader", "designer")
    # An incomplete design after the second search ends the search. An accepted candidate is still made.
    g.add_conditional_edges("designer", lambda s: "architect" if not s["missing"]
                            else "retriever" if s["round"] < 2 else "fabricator" if accepted(s) else "reporter")
    g.add_conditional_edges("engineer", lambda s: "tools" if s["messages"][-1].get("tool_calls")
                            and len(s["messages"]) < 12 else "judge")
    g.add_conditional_edges("judge", after_judge)
    g.add_edge("architect", "geometry")
    g.add_edge("geometry", "engineer")
    g.add_edge("tools", "engineer")
    g.add_edge("fabricator", "reporter")
    g.add_edge("reporter", END)
    return g.compile()
