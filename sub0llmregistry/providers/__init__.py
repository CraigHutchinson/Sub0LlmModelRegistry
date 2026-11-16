"""Provider plugins: where models come from.

A provider turns a spec (the text after "name:" in `fetch hf:org/repo#file.gguf`) into an Offer: the files to download, their
sizes and hashes when the provider publishes them, the version folder they belong in, and the manifest's `source`. Fetching
is shared: every file goes to `<root>/_incoming/` (resumable), is checked against its size and sha256, and only then is moved
into its version folder with a manifest, so a version folder never holds a partial download.

Plugins: the built-ins below, plus any installed package that declares an entry point in the group
"sub0llmregistry.providers" naming a Provider subclass (or instance):

    [project.entry-points."sub0llmregistry.providers"]
    myhub = "mypackage.myhub:MyHub"

Nothing is downloaded without the owner's go-ahead: `fetch` lists the files and total size and asks, unless --yes is given.
"""
import hashlib
import os
import shutil
import sys
import time
import urllib.request

USER_AGENT = "sub0llmregistry/0.1"
GROUP = "sub0llmregistry.providers"


class Offer:
    def __init__(self, folder, files, source, fmt="gguf", notes=""):
        self.folder = folder  # version folder relative to the root, "/" separated: <area>/.../<version>
        self.files = files  # [{"name", "url", "bytes" or None, "sha256" or None, "role" optional}]
        self.source = source  # manifest "source"
        self.format = fmt
        self.notes = notes
        self.headers = {}  # sent with every download (for example authorization read from the environment)
        self.pinned = False  # the folder names an immutable upstream revision: later fetches may add more of its files


class Provider:
    name = ""
    description = ""

    def offer(self, spec):
        """spec -> Offer. Reads the provider's index only; downloads nothing."""
        raise NotImplementedError

    def refetch(self, manifest):
        """A manifest this provider wrote -> its files again: [{"name", "url", "bytes", "sha256"}]. Providers that implement
        this let the registry evict their versions (cache.py) and restore them on request."""
        raise NotImplementedError


def available():
    from . import hf, url
    out = {p.name: p for p in (hf.HuggingFace(), url.Url())}
    try:
        from importlib.metadata import entry_points
        for ep in entry_points(group=GROUP):
            try:
                obj = ep.load()
                p = obj() if isinstance(obj, type) else obj
                out[p.name or ep.name] = p
            except Exception as e:  # a broken plugin must not break the registry
                print("provider %s failed to load: %s" % (ep.name, e), file=sys.stderr)
    except Exception:
        pass
    return out


def get(name):
    ps = available()
    if name not in ps:
        raise SystemExit("no provider %r (installed: %s)" % (name, ", ".join(sorted(ps))))
    return ps[name]


def request(url, headers=None, method="GET"):
    h = {"User-Agent": USER_AGENT}
    h.update(headers or {})
    return urllib.request.Request(url, headers=h, method=method)


def download(f, dest, headers=None, say=print):
    """Download f["url"] to dest, resuming a partial file; check size and sha256 when known. Returns the sha256."""
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if f.get("bytes") and have > f["bytes"]:
        os.remove(dest)
        have = 0
    if not (f.get("bytes") and have == f["bytes"]):
        h = dict(headers or {})
        if have:
            h["Range"] = "bytes=%d-" % have
        with urllib.request.urlopen(request(f["url"], h)) as resp:
            if have and resp.status != 206:  # the server ignored the range: start again
                have = 0
            total = f.get("bytes") or (int(resp.headers.get("Content-Length", 0)) + have) or None
            t0, last, done = time.time(), 0, have
            with open(dest, "ab" if have else "wb") as out:
                while True:
                    chunk = resp.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 5:
                        last = time.time()
                        rate = (done - have) / max(last - t0, 1e-6)
                        say("  %s: %.2f / %s GiB, %.0f MiB/s" % (f["name"], done / 2 ** 30, "%.2f" % (total / 2 ** 30) if total else "?", rate / 2 ** 20))
    size = os.path.getsize(dest)
    if f.get("bytes") and size != f["bytes"]:
        raise IOError("%s: got %d bytes, expected %d (kept in _incoming; run fetch again to resume)" % (f["name"], size, f["bytes"]))
    hsh = hashlib.sha256()
    with open(dest, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            hsh.update(chunk)
    digest = hsh.hexdigest()
    if f.get("sha256") and digest != f["sha256"]:
        os.remove(dest)
        raise IOError("%s: sha256 mismatch (deleted; run fetch again)" % f["name"])
    return digest


def fetch(reg, offer, say=print):
    """Download an offer into its version folder and write the manifest. Returns the manifest."""
    from ..store import MANIFEST, roleOf
    final = os.path.join(reg.root, *offer.folder.split("/"))
    old = reg.get(reg.idFor(final)) if os.path.exists(os.path.join(final, MANIFEST)) else None
    if old and not offer.pinned:
        raise SystemExit("already in the registry: %s" % reg.idFor(final))
    work = os.path.join(reg.root, "_incoming", offer.folder.replace("/", "__"))
    os.makedirs(work, exist_ok=True)
    files = []
    for f in offer.files:
        dest = os.path.join(work, *f["name"].split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        say("fetching %s" % f["name"])
        digest = download(f, dest, offer.headers, say)
        files.append({"name": f["name"], "bytes": os.path.getsize(dest), "sha256": digest, "role": f.get("role") or roleOf(f["name"])})
    os.makedirs(os.path.dirname(final), exist_ok=True)
    if os.path.exists(final):  # an earlier, unmanifested folder: move the files in beside whatever is there
        for f in files:
            dst = os.path.join(final, *f["name"].split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(os.path.join(work, *f["name"].split("/")), dst)
        shutil.rmtree(work, ignore_errors=True)
    else:
        shutil.move(work, final)
    import datetime
    from ..store import SCHEMA
    if old:  # more files of the same pinned upstream revision: they join its version (the revision itself never changes)
        names = {f["name"] for f in files}
        old["files"] = sorted([f for f in old["files"] if f["name"] not in names] + files, key=lambda x: x["name"])
        old.pop("facts", None)
        m = old
    else:
        m = {"schema": SCHEMA, "id": reg.idFor(final), "format": offer.format, "files": sorted(files, key=lambda x: x["name"]),
             "source": offer.source, "derivedFrom": None, "tool": None, "created": datetime.date.today().isoformat(), "notes": offer.notes}
    reg.writeManifest(final, m)
    return m


def confirm(say=print, question="download? [y/N] "):
    if not sys.stdin.isatty():
        say("not downloading: confirm with --yes (or run interactively)")
        return False
    try:
        answer = input(question)
    except EOFError:
        answer = ""
    return answer.strip().lower() in ("y", "yes")


def adopt(reg, src, source, alias=None, link=True, hashFiles=True, say=print):
    """Move an existing download (a folder, or a single file) into the registry as the version a provider names.

    Every offered file must be present in `src` with the provider's size, and (unless hashFiles is False, for files already
    checked) its sha256. The files are moved (a rename on the same drive), `src` is replaced by a symbolic link to the
    version folder (link=False: removed when empty), and MIGRATION.log gets a line."""
    import datetime
    from ..store import MANIFEST, SCHEMA, roleOf, sha256
    name, _, spec = source.partition(":")
    offer = get(name).offer(spec)
    src = os.path.abspath(src)
    srcDir = src if os.path.isdir(src) else os.path.dirname(src)
    final = os.path.join(reg.root, *offer.folder.split("/"))
    if os.path.exists(os.path.join(final, MANIFEST)):
        raise SystemExit("already in the registry: %s" % reg.idFor(final))
    files = []
    for f in offer.files:
        p = os.path.join(srcDir, *f["name"].split("/"))
        if not os.path.isfile(p):
            raise SystemExit("adopt: %s is not in %s" % (f["name"], srcDir))
        if f.get("bytes") and os.path.getsize(p) != f["bytes"]:
            raise SystemExit("adopt: %s is %d bytes, %s publishes %d" % (f["name"], os.path.getsize(p), name, f["bytes"]))
        digest = f.get("sha256")
        if hashFiles or not digest:
            say("hashing %s" % f["name"])
            digest = sha256(p)
            if f.get("sha256") and digest != f["sha256"]:
                raise SystemExit("adopt: %s sha256 differs from %s's (not adopted)" % (f["name"], name))
        files.append({"name": f["name"], "bytes": os.path.getsize(p), "sha256": digest, "role": f.get("role") or roleOf(f["name"])})
    os.makedirs(final, exist_ok=True)
    for f in files:
        dst = os.path.join(final, *f["name"].split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(os.path.join(srcDir, *f["name"].split("/")), dst)
    m = {"schema": SCHEMA, "id": reg.idFor(final), "format": offer.format, "files": sorted(files, key=lambda x: x["name"]),
         "source": offer.source, "derivedFrom": None, "tool": None, "created": datetime.date.today().isoformat(),
         "notes": "adopted from %s" % srcDir}
    reg.writeManifest(final, m)
    linked = False
    if not os.listdir(srcDir):
        os.rmdir(srcDir)
        if link:
            try:
                os.symlink(os.path.relpath(final, os.path.dirname(srcDir)), srcDir, target_is_directory=True)  # relative: survives remounts
                linked = True
            except OSError as e:
                say("no link left at %s: %s" % (srcDir, e))
    else:
        say("%s still holds other files; left in place without a link" % srcDir)
    with open(os.path.join(reg.root, "MIGRATION.log"), "a", encoding="utf-8", newline="\n") as log:
        log.write("%s\t%s\t-> %s%s\n" % (datetime.datetime.now().isoformat(timespec="seconds"), srcDir, m["id"],
                                          "\t(link left)" if linked else ""))
    try:
        reg.scanFacts(m["id"], say=say)
    except Exception as e:
        say("facts not read: %s" % e)
    if alias:
        reg.setAlias(alias, m["id"])
    reg.index()
    say("adopted %s" % m["id"])
    return m


def fetchCli(reg, source, yes=False, alias=None, say=print):
    name, _, spec = source.partition(":")
    if not spec:
        raise SystemExit("give provider:spec, e.g. hf:org/repo#file.gguf")
    offer = get(name).offer(spec)
    from ..store import MANIFEST
    final = os.path.join(reg.root, *offer.folder.split("/"))
    if os.path.exists(os.path.join(final, MANIFEST)):
        rid = reg.idFor(final)
        if reg.get(rid).get("evicted"):
            from ..cache import restore
            restore(reg, rid, yes=yes, say=say)
            if alias:
                reg.setAlias(alias, rid)
            reg.index()
            return 0
        have = {f["name"] for f in reg.get(rid)["files"]}
        wanted = [f["name"] for f in offer.files]
        offer.files = [f for f in offer.files if f["name"] not in have]
        if not offer.files or not offer.pinned:
            say("already in the registry: %s" % rid)
            if alias:
                reg.setAlias(alias, rid + ("#" + wanted[0] if len(wanted) == 1 else ""))
                reg.index()
            return 0
        say("%s is in the registry; adding files of the same revision:" % rid)
    total = sum(f.get("bytes") or 0 for f in offer.files)
    say("into %s/%s:" % (reg.root, offer.folder))
    for f in offer.files:
        say("  %-64s %s%s" % (f["name"], "%.2f GiB" % (f["bytes"] / 2 ** 30) if f.get("bytes") else "size unknown",
                              "" if f.get("sha256") else "  (no published sha256)"))
    say("total %.2f GiB from %s" % (total / 2 ** 30, offer.source.get("url") or offer.source.get("repo")))
    if not yes and not confirm(say):
        say("cancelled")
        return 1 if sys.stdin.isatty() else 2
    from ..cache import config, evict, plan, touch
    cfg = config(reg)
    if cfg.get("autoEvict"):
        for c in plan(reg, cfg=cfg, incoming=total):
            evict(reg, c[0], say)
    m = fetch(reg, offer, say)
    touch(reg, m["id"])
    try:
        reg.scanFacts(m["id"], say=say)
    except Exception as e:
        say("facts not read: %s" % e)
    if alias:
        reg.setAlias(alias, m["id"])
    reg.index()
    say("added %s" % m["id"])
    return 0
