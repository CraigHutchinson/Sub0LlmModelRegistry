"""python -m sub0llmregistry [--root DIR] <command> ...  (the registry root defaults to $SUB0_MODEL_REGISTRY, else D:/LlmModels)"""
import argparse
import datetime
import json
import shutil
import sys

from . import cache, providers
from .store import Evicted, Registry, roleOf


def gib(n):
    return "%.1f GiB" % (n / 2 ** 30)


def size(m):
    return sum(f.get("bytes", 0) for f in m["files"] if (f.get("role") or roleOf(f["name"])) in ("model", "shard"))


def ago(t):
    d = (datetime.datetime.now().timestamp() - t) / 86400
    return "today" if d < 1 else "%dd ago" % d


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
    r.add_argument("--restore", action="store_true", default=None, help="fetch an evicted version again first (no prompt)")
    r.add_argument("--no-track", action="store_true", help="do not record this as a use")
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
    st = sub.add_parser("status", help="disk use, cache policy, and versions by last use")
    st.add_argument("--json", action="store_true")
    ev = sub.add_parser("evict", help="delete least recently used downloads (manifests stay; restore fetches them again)")
    ev.add_argument("refs", nargs="*", help="evict exactly these (alias or id); otherwise by policy")
    ev.add_argument("--max-gib", type=float, help="shrink the registry to this size (default config.json cache.maxBytes)")
    ev.add_argument("--min-free-gib", type=float, help="leave this much free on the drive (default cache.minFreeBytes)")
    ev.add_argument("--idle-days", type=float, help="only versions unused this long (default cache.minIdleDays)")
    ev.add_argument("--dry-run", action="store_true")
    ev.add_argument("--yes", action="store_true")
    rs = sub.add_parser("restore", help="fetch an evicted version's files again (checked against its sha256)")
    rs.add_argument("ref")
    rs.add_argument("--yes", action="store_true")
    ad = sub.add_parser("adopt", help="move an existing download into the registry, checked against its provider")
    ad.add_argument("src", help="the folder (or file) holding the download")
    ad.add_argument("source", help="provider:spec it came from, e.g. hf:unsloth/Qwen3-8B-GGUF#Qwen3-8B-Q4_K_M.gguf")
    ad.add_argument("--alias")
    ad.add_argument("--no-link", action="store_true", help="do not leave a symbolic link at the old path")
    ad.add_argument("--no-hash", action="store_true", help="trust sizes and record the provider's sha256 (files already checked)")
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
        u = cache.usage(reg)
        for m in ms:
            fa = m.get("facts") or {}
            desc = " ".join(str(x) for x in (fa.get("architecture"), fa.get("sizeLabel"), fa.get("fileType")) if x)
            print("%-64s %9s %-8s %-8s %-9s %s%s" % (m["id"], gib(size(m)), "evicted" if m.get("evicted") else "", m.get("format", ""),
                                                  ago(cache.lastUsed(reg, m, u)), desc,
                                                  "  [" + ", ".join(back[m["id"]]) + "]" if m["id"] in back else ""))
    elif args.cmd == "show":
        print(json.dumps(reg.get(reg.aliases().get(args.ref, args.ref).partition("#")[0]), indent=2))
    elif args.cmd == "resolve":
        try:
            path, rid = reg.resolve(args.ref, restore=args.restore, track=not args.no_track, say=lambda *a: print(*a, file=sys.stderr))
        except Evicted as e:
            print(e, file=sys.stderr)
            return 3
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
    elif args.cmd == "status":
        cfg = cache.config(reg)
        keep = cache.keepSet(reg, cfg)
        u = cache.usage(reg)
        rows = []
        for m in reg.manifests():
            rows.append({"id": m["id"], "onDisk": cache.onDisk(reg, m), "bytes": sum(f["bytes"] for f in m["files"]),
                         "lastUsed": cache.lastUsed(reg, m, u), "uses": (u.get(m["id"]) or {}).get("uses", 0),
                         "evicted": bool(m.get("evicted")), "evictable": cache.whyNot(reg, m, keep)})
        rows.sort(key=lambda r: r["lastUsed"])
        du = shutil.disk_usage(reg.root)
        if args.json:
            print(json.dumps({"root": reg.root, "cache": cfg, "free": du.free, "versions": rows}, indent=1))
            return 0
        lim = lambda v: gib(v) if v else "none"
        print("%s: %s on disk, %s free; %d versions, %d evicted (%s not on disk)" % (
            reg.root, gib(sum(r["onDisk"] for r in rows)), gib(du.free), len(rows), sum(r["evicted"] for r in rows),
            gib(sum(r["bytes"] for r in rows if r["evicted"]))))
        print("policy: maxBytes %s, minFreeBytes %s, minIdleDays %s, autoEvict %s, autoRestore %s, keep %s" % (
            lim(cfg["maxBytes"]), lim(cfg["minFreeBytes"]), cfg["minIdleDays"], cfg["autoEvict"], cfg["autoRestore"], sorted(keep) or "[]"))
        print("least recently used first:")
        for r in rows:
            state = "evicted" if r["evicted"] else ("evictable" if not r["evictable"] else "kept: " + r["evictable"])
            print("  %-64s %9s  %-9s %3d uses  %s" % (r["id"], gib(r["bytes"]), ago(r["lastUsed"]), r["uses"], state))
    elif args.cmd == "evict":
        g = lambda v: None if v is None else int(v * 2 ** 30)
        todo = cache.plan(reg, refs=args.refs, maxBytes=g(args.max_gib), minFreeBytes=g(args.min_free_gib), idleDays=args.idle_days)
        if not todo:
            print("nothing to evict")
            return 0
        for m, n, t in todo:
            print("  %-64s %9s  last used %s" % (m["id"], gib(n), ago(t)))
        print("evict %d version(s), %s (manifests stay; `restore` fetches them again)" % (len(todo), gib(sum(c[1] for c in todo))))
        if args.dry_run:
            return 0
        if not args.yes and not providers.confirm(question="evict? [y/N] "):
            print("cancelled")
            return 1
        for m, _, _ in todo:
            cache.evict(reg, m)
        reg.index()
    elif args.cmd == "restore":
        cache.restore(reg, reg.aliases().get(args.ref, args.ref).partition("#")[0], yes=args.yes)
        reg.index()
    elif args.cmd == "adopt":
        providers.adopt(reg, args.src, args.source, alias=args.alias, link=not args.no_link, hashFiles=not args.no_hash)
    elif args.cmd == "fetch":
        return providers.fetchCli(reg, args.source, yes=args.yes, alias=args.alias)
    return 0


if __name__ == "__main__":
    sys.exit(main())
