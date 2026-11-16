"""Disk use: which versions were used when, evicting the least recently used downloads, and restoring them on request.

An evicted version keeps its folder and MANIFEST.json (source, sizes, sha256 and facts) and loses its files; the manifest
records `"evicted": {"at", "bytes"}`. Restoring downloads the files again from the manifest's `source` and accepts them
only if every sha256 matches, so a restored version is byte-for-byte the version that was evicted.

Only versions that can be fetched again are evicted: a download area whose provider implements `refetch`, with a sha256
recorded for every file. What we made (`local/`, `sub0llm/`) is never evicted. See spec/LAYOUT.md "Disk use".
"""
import datetime
import json
import os
import re
import shutil
import time

USAGE = "usage.json"
CONFIG = "config.json"
DEFAULTS = {"maxBytes": None, "minFreeBytes": None, "minIdleDays": 14, "keep": [], "autoRestore": False, "autoEvict": False}


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def config(reg):
    """The registry's cache policy: config.json "cache" over DEFAULTS."""
    out = dict(DEFAULTS)
    p = os.path.join(reg.root, CONFIG)
    if os.path.exists(p):
        with open(p, encoding="utf-8-sig") as f:
            out.update((json.load(f) or {}).get("cache") or {})
    return out


# ---- usage -------------------------------------------------------------------------------------------------------------------
def usage(reg):
    p = os.path.join(reg.root, USAGE)
    try:
        with open(p, encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def touch(reg, rid):
    """Record that a version was used. Best effort: a read-only registry or a lost race costs one timestamp, nothing else."""
    try:
        u = usage(reg)
        e = u.setdefault(rid, {})
        e["lastUsed"] = now()
        e["uses"] = e.get("uses", 0) + 1
        tmp = os.path.join(reg.root, "%s.%d.tmp" % (USAGE, os.getpid()))
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(dict(sorted(u.items())), f, indent=1)
            f.write("\n")
        os.replace(tmp, os.path.join(reg.root, USAGE))
    except OSError:
        pass


def lastUsed(reg, m, u=None):
    """When a version was last used (epoch seconds): the latest of usage.json, the files' access times (readers that open
    paths without the tool; coarse under relatime) and the manifest's `created` date."""
    from .store import filePath
    u = usage(reg) if u is None else u
    t = []
    if (u.get(m["id"]) or {}).get("lastUsed"):
        t.append(datetime.datetime.fromisoformat(u[m["id"]]["lastUsed"]).timestamp())
    if not m.get("evicted"):
        for f in m["files"]:
            try:
                t.append(os.stat(filePath(reg, m, f["name"])).st_atime)
            except OSError:
                pass
    try:
        t.append(datetime.datetime.fromisoformat(m.get("created", "")).timestamp())
    except ValueError:
        pass
    if (m.get("evicted") or {}).get("at"):
        t.append(datetime.datetime.fromisoformat(m["evicted"]["at"]).timestamp())
    return max(t) if t else 0.0


def onDisk(reg, m):
    from .store import filePath
    n = 0
    for f in m["files"]:
        try:
            n += os.path.getsize(filePath(reg, m, f["name"]))
        except OSError:
            pass
    return n


# ---- eviction ----------------------------------------------------------------------------------------------------------------
def whyNot(reg, m, keep=()):
    """None when the version can be evicted, else the reason it cannot."""
    from . import providers
    if m.get("evicted"):
        return "already evicted"
    if m["id"] in keep:
        return "kept (config.json cache.keep)"
    src = m.get("source") or {}
    prov = src.get("provider") or src.get("kind")
    area = m["id"].split("/", 1)[0]
    if area in ("local", "sub0llm") or not prov:
        return "made here, cannot be fetched again"
    p = providers.available().get(prov)
    if not p or type(p).refetch is providers.Provider.refetch:
        return "provider %r cannot fetch it again" % prov
    missing = [f["name"] for f in m["files"] if not f.get("sha256")]
    if missing:
        return "no sha256 recorded for %s (run verify --hash)" % ", ".join(missing)
    return None


def keepSet(reg, cfg):
    al = reg.aliases()
    return {al.get(k, k).partition("#")[0] for k in cfg.get("keep") or []}


def plan(reg, refs=None, maxBytes=None, minFreeBytes=None, idleDays=None, cfg=None, incoming=0):
    """The versions to evict, least recently used first, as [(manifest, bytes on disk, last used)].

    With `refs`, exactly those (each must be evictable). Otherwise the least recently used evictable versions idle at least
    `idleDays` (default config minIdleDays) until the registry fits `maxBytes` and leaves `minFreeBytes` free, after making
    room for `incoming` bytes; with neither limit, every evictable version idle that long."""
    cfg = cfg or config(reg)
    keep = keepSet(reg, cfg)
    u = usage(reg)
    if refs:
        out = []
        for ref in refs:
            m = reg.get(reg.aliases().get(ref, ref).partition("#")[0])
            why = whyNot(reg, m, keep)
            if why:
                raise SystemExit("%s: not evicted: %s" % (m["id"], why))
            out.append((m, onDisk(reg, m), lastUsed(reg, m, u)))
        return out
    maxBytes = maxBytes if maxBytes is not None else cfg.get("maxBytes")
    minFreeBytes = minFreeBytes if minFreeBytes is not None else cfg.get("minFreeBytes")
    idle = (idleDays if idleDays is not None else cfg.get("minIdleDays") or 0) * 86400
    ms = reg.manifests()
    total = sum(onDisk(reg, m) for m in ms) + incoming
    free = shutil.disk_usage(reg.root).free - incoming
    cands = []
    for m in ms:
        if whyNot(reg, m, keep):
            continue
        t = lastUsed(reg, m, u)
        if time.time() - t >= idle:
            cands.append((m, onDisk(reg, m), t))
    cands.sort(key=lambda c: c[2])
    if maxBytes is None and minFreeBytes is None:
        return cands
    out = []
    for c in cands:
        if (maxBytes is None or total <= maxBytes) and (minFreeBytes is None or free >= minFreeBytes):
            break
        out.append(c)
        total -= c[1]
        free += c[1]
    return out


def evict(reg, m, say=print):
    """Delete a version's files, keeping its folder and manifest. Returns the bytes freed."""
    from .store import filePath
    why = whyNot(reg, m)
    if why:
        raise SystemExit("%s: not evicted: %s" % (m["id"], why))
    folder = reg.folderFor(m["id"])
    freed = 0
    for f in m["files"]:
        p = filePath(reg, m, f["name"])
        if os.path.exists(p):
            freed += os.path.getsize(p)
            os.remove(p)
        d = os.path.dirname(p)  # sub-folders left empty
        while os.path.normcase(d) != os.path.normcase(folder) and os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
            d = os.path.dirname(d)
    m["evicted"] = {"at": now(), "bytes": freed}
    reg.writeManifest(folder, m)
    say("evicted %s (%.2f GiB)" % (m["id"], freed / 2 ** 30))
    return freed


def restore(reg, rid, yes=False, track=True, say=print):
    """Fetch an evicted version's files again from its source; each must match its recorded sha256."""
    from . import providers
    m = reg.get(rid)
    if not m.get("evicted"):
        return m
    src = m.get("source") or {}
    p = providers.get(src.get("provider") or src.get("kind"))
    files = p.refetch(m)
    need = sum(f.get("bytes") or 0 for f in files)
    say("restoring %s: %d file(s), %.2f GiB from %s" % (m["id"], len(files), need / 2 ** 30, src.get("url") or src.get("repo")))
    if not yes and not providers.confirm(say):
        raise SystemExit("not restored: %s" % m["id"])
    cfg = config(reg)
    if cfg.get("autoEvict"):
        for c in plan(reg, cfg=cfg, incoming=need):
            if c[0]["id"] != m["id"]:
                evict(reg, c[0], say)
    if shutil.disk_usage(reg.root).free < need:
        raise SystemExit("not restored: %.2f GiB needed, %.2f GiB free (evict something first)"
                         % (need / 2 ** 30, shutil.disk_usage(reg.root).free / 2 ** 30))
    work = os.path.join(reg.root, "_incoming", m["path"].replace("/", "__"))
    os.makedirs(work, exist_ok=True)
    for f in files:
        dest = os.path.join(work, *re.split(r"[\\/]", f["name"]))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        say("fetching %s" % f["name"])
        providers.download(f, dest, getattr(p, "headers", lambda: {})(), say)
    folder = reg.folderFor(m["id"])
    for f in files:
        dst = os.path.join(folder, *re.split(r"[\\/]", f["name"]))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(os.path.join(work, *re.split(r"[\\/]", f["name"])), dst)
    shutil.rmtree(work, ignore_errors=True)
    m.pop("evicted", None)
    reg.writeManifest(folder, m)
    if track:
        touch(reg, m["id"])
    say("restored %s" % m["id"])
    return m
