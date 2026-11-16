"""python -m sub0llmregistry [--root DIR] <command> ...  (the registry root defaults to $SUB0_MODEL_REGISTRY, else D:/LlmModels)"""
import argparse
import json
import sys

from . import providers
from .store import Registry, roleOf


def gib(n):
    return "%.1f GiB" % (n / 2 ** 30)


def size(m):
    return sum(f.get("bytes", 0) for f in m["files"] if (f.get("role") or roleOf(f["name"])) in ("model", "shard"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sub0llmregistry", description=__doc__)
    ap.add_argument("--root", help="registry folder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("list", help="entries, optionally filtered")
    q.add_argument("text", nargs="?", help="substring of id, model name or notes")
    q.add_argument("--format")
    q.add_argument("--area")
    q.add_argument("--arch")
    q.add_argument("--max-gib", type=float)
    q.add_argument("--min-context", type=int)
    q.add_argument("--moe", action="store_true", default=None)
    q.add_argument("--json", action="store_true")
    s = sub.add_parser("show", help="one entry's manifest (alias or id)")
    s.add_argument("ref")
    r = sub.add_parser("resolve", help="alias / id / id#file -> absolute path")
    r.add_argument("ref")
    r.add_argument("--json", action="store_true")
    sub.add_parser("index", help="regenerate registry.json")
    v = sub.add_parser("verify", help="check files exist with their recorded sizes")
    v.add_argument("--hash", action="store_true", help="also check sha256 (recorded where missing; reads every byte)")
    f = sub.add_parser("facts", help="read GGUF headers into manifests")
    f.add_argument("ref", nargs="?")
    f.add_argument("--force", action="store_true")
    n = sub.add_parser("new-manifest", help="describe a version folder already in place")
    n.add_argument("folder")
    n.add_argument("--format", default="gguf")
    n.add_argument("--provider")
    n.add_argument("--repo")
    n.add_argument("--revision")
    n.add_argument("--derived-from")
    n.add_argument("--notes", default="")
    a = sub.add_parser("alias", help="set (or with no target, remove) an alias")
    a.add_argument("name")
    a.add_argument("target", nargs="?")
    sub.add_parser("providers", help="installed provider plugins")
    fe = sub.add_parser("fetch", help="download from a provider: shows the files and sizes, then asks")
    fe.add_argument("source", help="provider:spec, e.g. hf:unsloth/Qwen3-8B-GGUF#Qwen3-8B-Q4_K_M.gguf or url:https://...")
    fe.add_argument("--yes", action="store_true", help="skip the confirmation (the owner already agreed to this download)")
    fe.add_argument("--alias")
    args = ap.parse_args(argv)
    reg = Registry(args.root)

    if args.cmd == "list":
        ms = reg.query(text=args.text, format=args.format, area=args.area, architecture=args.arch,
                       maxBytes=args.max_gib and int(args.max_gib * 2 ** 30), minContext=args.min_context, moe=args.moe)
        if args.json:
            print(json.dumps(ms, indent=1))
            return 0
        back = {}
        for k, t in reg.aliases().items():
            back.setdefault(t.partition("#")[0], []).append(k)
        for m in ms:
            fa = m.get("facts") or {}
            desc = " ".join(str(x) for x in (fa.get("architecture"), fa.get("sizeLabel"), fa.get("fileType")) if x)
            print("%-72s %9s  %-8s %s%s" % (m["id"], gib(size(m)), m.get("format", ""), desc,
                                            "  [" + ", ".join(back[m["id"]]) + "]" if m["id"] in back else ""))
    elif args.cmd == "show":
        print(json.dumps(reg.get(reg.aliases().get(args.ref, args.ref).partition("#")[0]), indent=2))
    elif args.cmd == "resolve":
        path, rid = reg.resolve(args.ref)
        print(json.dumps({"path": path, "id": rid}) if args.json else path)
    elif args.cmd == "index":
        entries, al, warn = reg.index()
        for w in warn:
            print("warning:", w, file=sys.stderr)
        print("%d entries, %d aliases -> %s" % (len(entries), len(al), reg.root))
    elif args.cmd == "verify":
        bad = reg.verify(hashes=args.hash)
        print("ok" if not bad else "%d problem(s)" % bad)
        return 1 if bad else 0
    elif args.cmd == "facts":
        print("%d manifest(s) updated" % reg.scanFacts(args.ref and reg.aliases().get(args.ref, args.ref).partition("#")[0], force=args.force))
    elif args.cmd == "new-manifest":
        src = {"provider": args.provider, "repo": args.repo, "revision": args.revision, "url": None} if args.provider else None
        m = reg.newManifest(args.folder, fmt=args.format, source=src, derivedFrom=args.derived_from, notes=args.notes)
        print(m["id"])
    elif args.cmd == "alias":
        reg.setAlias(args.name, args.target)
        reg.index()
        print("%s -> %s" % (args.name, args.target or "(removed)"))
    elif args.cmd == "providers":
        for name, p in sorted(providers.available().items()):
            print("%-8s %s" % (name, p.description))
    elif args.cmd == "fetch":
        return providers.fetchCli(reg, args.source, yes=args.yes, alias=args.alias)
    return 0


if __name__ == "__main__":
    sys.exit(main())
