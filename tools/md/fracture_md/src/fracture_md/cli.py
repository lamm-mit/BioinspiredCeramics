"""Command-line entry point ``fracture-md`` (thin wrapper over fracture_md.agents).

    fracture-md potentials                         # potential cards + domain status table
    fracture-md submit proposal.json [-o result.json] [--runs-dir runs] [--registry implementations]
    fracture-md impl [--root DIR] register --kind K --name N --source FILE --author A [--gate]
    fracture-md impl [--root DIR] gate <id> [--seed S] | list [--kind K] [--status S] | show <id>

``impl --root`` defaults to $FRACTURE_MD_IMPLEMENTATIONS, else the store ``submit`` uses by default
(``agents.interface.default_registry()``: ``<repo>/implementations``), so a registered and gated
implementation is visible to ``submit`` without extra flags.
    fracture-md schema ExperimentProposal|ExperimentResult|<any fracture_md.schema model>

Exit codes of ``submit``: 0 completed, 1 partial, 2 rejected, 3 failed.
``impl gate`` exits 0 if the implementation passed (trusted), 1 otherwise; ``impl`` exits 2 on a
missing, corrupt or tampered store entry (message on stderr).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

_EXIT = {"completed": 0, "partial": 1, "rejected": 2, "failed": 3}
_SHORT = {
    "validated": "OK",
    "partially_validated": "partial",
    "unvalidated": "unval.",
    "known_deficient": "DEFICIENT",
}


def _cmd_potentials(args) -> int:
    from fracture_md.potentials import REGISTRY
    from fracture_md.schema.potential import PropertyClass

    classes = [pc for pc in PropertyClass]
    rows = []
    for name, factory in REGISTRY.items():
        try:
            card = factory().card
        except ImportError as exc:  # optional potential (e.g. MACE-JAX) not installed here
            print(f"# {name}: unavailable ({str(exc).splitlines()[0][:120]})", file=sys.stderr)
            continue
        rows.append((name, card, None))
    if args.registry:
        from fracture_md.agents.registry import ImplementationRegistry

        reg = ImplementationRegistry(args.registry)
        for man in reg.list(kind="potential_energy_fn", status="trusted"):
            try:
                rows.append((f"impl:{man['id']}", reg.energy_model(man["id"]).card, man))
            except Exception as exc:  # noqa: BLE001
                print(f"# skipping {man['id']}: {exc}", file=sys.stderr)
    for name, card, _man in rows:
        print(f"{name}\n  id       {card.id}\n  name     {card.name}\n  kind     {card.kind.value}\n"
              f"  elements {','.join(card.elements)}   cutoff {card.cutoff.magnitude():.3f} {card.cutoff.unit.value}\n"
              f"  license  {card.license[:110]}")
    print()
    w = max(len(r[0]) for r in rows) + 2
    cw = max(len(pc.value) for pc in classes) + 1
    print("domain status per property class (DEFICIENT = known_deficient -> OUT_OF_DOMAIN records; "
          "submit() rejects such protocols unless allow_out_of_domain)")
    print("potential".ljust(w) + "".join(pc.value.ljust(cw) for pc in classes))
    for name, card, _ in rows:
        print(name.ljust(w) + "".join(_SHORT[card.status_for(pc)].ljust(cw) for pc in classes))
    return 0


def _cmd_submit(args) -> int:
    from fracture_md.agents.interface import default_registry, submit
    from fracture_md.agents.registry import ImplementationRegistry

    reg = ImplementationRegistry(args.registry) if args.registry else default_registry()
    res = submit(Path(args.proposal), runs_dir=args.runs_dir, registry=reg)
    text = res.model_dump_json(indent=1)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    summary = {
        "result_id": res.result_id,
        "status": res.status.value,
        "n_properties": len(res.properties),
        "criteria": [{"property": o.criterion.property_name, "observed": o.observed, "satisfied": o.satisfied}
                     for o in res.criteria],
        "warnings": res.warnings,
        "errors": res.errors,
        "run_dir": str(Path(args.runs_dir) / res.result_id) if args.runs_dir else None,
    }
    print(json.dumps(summary, indent=1) if args.output or args.summary else text)
    return _EXIT[res.status.value]


def _cmd_impl(args) -> int:
    from fracture_md.agents.registry import RegistryError, TamperError

    try:
        return _cmd_impl_inner(args)
    except (RegistryError, TamperError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _impl_root(args) -> str:
    if args.root:
        return args.root
    if os.environ.get("FRACTURE_MD_IMPLEMENTATIONS"):
        return os.environ["FRACTURE_MD_IMPLEMENTATIONS"]
    from fracture_md.agents.interface import default_registry

    return str(default_registry().root)


def _cmd_impl_inner(args) -> int:
    from fracture_md.agents.gates import gate
    from fracture_md.agents.registry import ImplementationRegistry

    reg = ImplementationRegistry(_impl_root(args))
    if args.impl_cmd == "register":
        iid = reg.register(args.kind, args.name, Path(args.source), author=args.author, parent_id=args.parent,
                           description=args.description, protocol_kind=args.protocol_kind,
                           base_potential=args.base_potential)
        if args.gate:
            rep = gate(reg, iid)
            print(rep.summary(), file=sys.stderr)
        print(iid)
        return 0
    if args.impl_cmd == "gate":
        rep = gate(reg, args.id, reference_dir=args.reference_dir, seed=args.seed)
        print(rep.summary())
        return 0 if rep.passed else 1
    if args.impl_cmd == "list":
        mans = reg.list(kind=args.kind, status=args.status)
        for m in mans:
            n = len(m.get("gate_results", []))
            nf = sum(1 for c in m.get("gate_results", []) if not c.get("passed"))
            trusted = "" if m.get("status") != "trusted" else ("" if reg.is_trusted(m["id"]) else " (NOT trusted: "
                                                                                                  "stale/incomplete gate)")
            print(f"{m['id']:<40} {m.get('kind', '?'):<20} {m.get('status', '?'):<9} gate {n - nf}/{n}  "
                  f"parent={m.get('parent_id')}  author={m.get('author')}{trusted}")
        if not mans:
            print(f"(no implementations under {reg.root})")
        return 0
    if args.impl_cmd == "show":
        man = reg.get_manifest(args.id)
        man["source_hash_ok"] = reg.verify(args.id)
        man["trust_problems"] = reg.trust_problems(args.id, man)
        print(json.dumps(man, indent=1))
        return 0
    return 2  # pragma: no cover


def _cmd_schema(args) -> int:
    import fracture_md.schema as schema

    model = getattr(schema, args.model, None)
    if model is None or not hasattr(model, "model_json_schema"):
        names = sorted(n for n in dir(schema) if hasattr(getattr(schema, n), "model_json_schema"))
        print(f"unknown model {args.model!r}; choose from {names}", file=sys.stderr)
        return 2
    print(json.dumps(model.model_json_schema(), indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fracture-md", description="JAX-first atomistic fracture harness (agent interface)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("potentials", help="list potential cards and their per-property-class domain status")
    sp.add_argument("--registry", default=None, help="also list trusted potential implementations from this store")
    sp.set_defaults(fn=_cmd_potentials)

    sp = sub.add_parser("submit", help="submit an ExperimentProposal JSON file")
    sp.add_argument("proposal")
    sp.add_argument("-o", "--output", help="write the full ExperimentResult JSON here (stdout gets a summary)")
    sp.add_argument("--runs-dir", default="runs")
    sp.add_argument("--registry", default=None, help="implementation store (default: <repo>/implementations)")
    sp.add_argument("--summary", action="store_true", help="print a summary instead of the full result")
    sp.set_defaults(fn=_cmd_submit)

    sp = sub.add_parser("impl", help="versioned implementation registry")
    sp.add_argument("--root", default=None,
                    help="implementation store (default: $FRACTURE_MD_IMPLEMENTATIONS, else the store submit uses, "
                         "<repo>/implementations)")
    isub = sp.add_subparsers(dest="impl_cmd", required=True)
    r = isub.add_parser("register")
    r.add_argument("--kind", required=True, choices=["potential_energy_fn", "protocol_runner"])
    r.add_argument("--name", required=True)
    r.add_argument("--source", required=True)
    r.add_argument("--author", required=True)
    r.add_argument("--parent", default=None)
    r.add_argument("--description", default="")
    r.add_argument("--protocol-kind", default=None)
    r.add_argument("--base-potential", default=None)
    r.add_argument("--gate", action="store_true", help="run the regression gate right after registering")
    g = isub.add_parser("gate")
    g.add_argument("id")
    g.add_argument("--reference-dir", default=None)
    g.add_argument("--seed", type=int, default=None, help="seed of the random gate part (default: fresh)")
    lst = isub.add_parser("list")
    lst.add_argument("--kind", default=None)
    lst.add_argument("--status", default=None)
    sh = isub.add_parser("show")
    sh.add_argument("id")
    sp.set_defaults(fn=_cmd_impl)

    sp = sub.add_parser("schema", help="print the JSON schema of a model (e.g. ExperimentProposal)")
    sp.add_argument("model")
    sp.set_defaults(fn=_cmd_schema)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    if os.environ.get("FRACTURE_MD_ALLOW_GPU") != "1":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import fracture_md  # noqa: F401  (float64)

    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
