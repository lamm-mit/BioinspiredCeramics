"""The agent graph. Each node is one agent with its own model and instructions.

planner -> retriever -> figure_reader -> designer -> geometry -> engineer <-> tools -> reporter

The designer can send the retriever back once, for design fields that lack evidence. A check
compares each design value with the source the designer cites, and the designer gets one repair
with its findings. A field that still fails is dropped. A design that is still incomplete goes
straight to the reporter, so no simulation runs on it. A complete design gets its geometry as
STL files. The engineer then calls tools until it has every result, and each call is recorded
in one handoff file per tool. The prompts are generic, and the problem itself is defined by
design.py and the tools package.
"""

import json
import operator
import re
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from tools import RUN, TOOLS

from . import corpus as cp
from . import handoff
from .design import ASK, DESIGN, RANGE, SCALE, UNIT, design_errors, make_geometry
from .llm import ask_json, chat, user

# Model and reasoning effort per agent. A text model does the four text jobs and a vision model
# reads the figures, with thinking off.
MODELS = {"planner": ("gpt-oss:20b", "low"),
          "figure_reader": ("qwen3.6:35b", "none"),
          "designer": ("gpt-oss:20b", "medium"),
          "checker": ("gpt-oss:20b", "medium"),
          "engineer": ("gpt-oss:20b", "low"),
          "reporter": ("gpt-oss:20b", "low")}
EMBEDDING = "embeddinggemma"
FIELDS = json.dumps(DESIGN)
OPTIONS = sorted({option for field in DESIGN.values() for option in field.get("enum", ())})

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

ENGINEER = """You run the simulations for a design with the tools.
Take every tool input from the design, the geometry or an earlier tool result, as the tool
descriptions say. Use no other numbers. If an input is available from none of them, do not call
that tool and say which input is missing. When every tool has run, reply in two sentences and call no tool."""

CHECKER = """You check one value of a design against its source text. Answer from the source text only.
Return JSON {"low": number or null, "high": number or null, "unit": str} with the smallest and the largest value
the source gives for the quantity asked, in the unit asked. For a single value, low equals high. Use null for both
when the source gives no value for that quantity. Reply with the JSON only."""

REPORTER = """Write a short report in plain markdown, without emoji, for a scientist.
Give the question, a table of the design fields with value, unit and source, the simulation
results, what they mean, and what remains unverified. Use only numbers and criteria that appear
in the input. Results whose method says stand-in are placeholders, not predictions. Say so.
If design fields are missing, no simulation ran. Say which fields are missing, and give the
reason for each field that the source check rejected."""


class State(TypedDict, total=False):
    question: str
    folder: str
    queries: list
    entities: list
    round: int
    passages: Annotated[list, operator.add]
    triples: Annotated[list, operator.add]
    figure_notes: Annotated[list, operator.add]
    design: dict
    sources: dict
    missing: list
    rejected: dict
    candidate: str
    geometry: dict
    messages: list
    results: list
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
    return rows + [(t["source"], f"{t['head']} -> {t['relation']} -> {t['tail']}") for t in s["triples"]]


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
            if not (any(low - slack <= value <= high + slack for low, high in stated)
                    or field in RANGE and spans and min(spans)[0] <= value <= max(high for _, high in spans)):
                unit = next(u for u, factor in SCALE[field].items() if factor == 1)
                values = ", ".join(f"{low:g}" if low == high else f"{low:g} to {high:g}" for low, high in sorted(set(stated))[:12])
                found = f"gives {values} {unit}, not {value}." if stated else f"gives no value in {unit}."
                errors[field] = f"{' and '.join(ids)} {found}{hint}"
    return errors


def build(models, corpus):
    def ask(agent, system, text, images=()):
        model, effort = models[agent]
        try:
            out, rec = ask_json(model, system, text, images, effort)
        except ValueError:
            # A thinking model can spend its whole token budget on reasoning and return no answer.
            # It gets one more try at low effort.
            if effort in (None, "none", "low"):
                raise
            out, rec = ask_json(model, system, text, images, "low")
            rec["retried"] = True
        return out, rec | {"agent": agent}

    def call(agent, messages, tools=None):
        model, effort = models[agent]
        reply, rec = chat(model, messages, tools, effort)
        return reply, rec | {"agent": agent}

    def planner(s):
        out, rec = ask("planner", PLANNER, f"Question: {s['question']}\nDesign fields: {FIELDS}")
        return {"queries": out["queries"], "entities": out["entities"], "round": 0, "trace": [rec]}

    def retriever(s):
        seen = {p["id"] for p in s["passages"]}
        found = [p for q in s["queries"]
                 for p in cp.search(corpus, q, 4, skip=seen) + cp.search(corpus, q, 1, figures=True, skip=seen)]
        # Semantic search favors passages on the topic over passages that state a value or name a
        # material. So an open field is also searched among the passages that can supply it. Those
        # hold a number with the field's unit, or the name of one of its options. The first round
        # does this for the numeric fields, and the second for every field still missing.
        for field in s.get("missing", UNIT):
            query = f"{s['question']} {DESIGN[field]['description']}"
            if field in UNIT:
                found += cp.search(corpus, query, 4, skip=seen, pattern=UNIT[field])
            for option in DESIGN[field].get("enum", ()):
                found += cp.search(corpus, query, 1, skip=seen, pattern=f"(?i){re.escape(option)}")
        # The knowledge graph is asked about the planner's entities and about every allowed design option.
        triples = cp.neighbors(corpus, s["entities"] + OPTIONS) if s["round"] == 0 else []
        return {"passages": list({p["id"]: p for p in found}.values()), "triples": triples, "round": s["round"] + 1}

    def figure_reader(s):
        done = {f["id"] for f in s["figure_notes"]}
        notes, trace = [], []
        for p in s["passages"]:
            if "image" in p and p["id"] not in done:
                out, rec = ask("figure_reader", FIGURE_READER, f"Caption: {p['text']}\nDesign fields: {FIELDS}",
                               [corpus["folder"] / p["image"]])
                out["values"] = [v for v in out.get("values") or [] if reading(v)]
                notes.append({"id": p["id"], "caption": p["text"], "reading": out})
                trace.append(rec | {"figure": p["id"]})
        return {"figure_notes": notes, "trace": trace}

    def designer(s):
        rows = shown(s)
        prompt = (f"Question: {s['question']}\nDesign fields: {FIELDS}\nEvidence:\n"
                  + "\n".join(f"[E{n}] {text}" for n, (_, text) in enumerate(rows, 1)))

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
            for field in ASK.keys() & design.keys():
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

        design, sources, rejected, trace, read = {}, {}, {}, [], {}
        for attempt in range(2):  # the second attempt is the one repair, with the findings of the source check
            out, rec = ask("designer", DESIGNER, prompt)
            trace.append(rec)
            # A repair may answer only the rejected fields, so each answer adds to the fields that passed.
            design |= {k: v for k, v in (out.get("design") or {}).items() if k in DESIGN and v is not None}
            sources |= {k: real(str(v)) for k, v in (out.get("sources") or {}).items()}
            failed = source_errors(design, sources, s) | design_errors(design)
            failed |= second_reading({k: v for k, v in design.items() if k not in failed})
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

    def geometry(s):
        """Start a candidate: its folder, the design with its sources, and the STL files of the part."""
        folder = Path(s["folder"]) / "candidate_1"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "design.json").write_text(json.dumps({"design": s["design"], "sources": s["sources"]}, indent=1, ensure_ascii=False))
        return {"candidate": str(folder), "geometry": make_geometry(s["design"], folder / "geometry"), "messages": [], "results": []}

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

    def reporter(s):
        facts = {k: s[k] for k in ["question", "design", "sources", "missing", "rejected", "geometry", "results"] if k in s}
        reply, rec = call("reporter", [{"role": "system", "content": REPORTER}, user(json.dumps(facts))])
        return {"report": reply["content"], "trace": [rec]}

    g = StateGraph(State)
    for node in [planner, retriever, figure_reader, designer, geometry, engineer, tools, reporter]:
        g.add_node(node.__name__, node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "figure_reader")
    g.add_edge("figure_reader", "designer")
    g.add_conditional_edges("designer", lambda s: "geometry" if not s["missing"]
                            else "retriever" if s["round"] < 2 else "reporter")
    g.add_conditional_edges("engineer", lambda s: "tools" if s["messages"][-1].get("tool_calls")
                            and len(s["messages"]) < 12 else "reporter")
    g.add_edge("geometry", "engineer")
    g.add_edge("tools", "engineer")
    g.add_edge("reporter", END)
    return g.compile()
