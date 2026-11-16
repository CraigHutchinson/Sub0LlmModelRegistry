"""A registry folder (spec/LAYOUT.md): manifests, ids, aliases, the generated index, queries and verification."""
import datetime
import hashlib
import json
import os
import re

MANIFEST = "MANIFEST.json"
SCHEMA = "sub0llm-model-registry/manifest/1"
SKIP_DIRS = {"_incoming", ".git"}
SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def userConfigPath():
    base = os.environ.get("APPDATA") if os.name == "nt" else os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base or os.path.expanduser("~"), "sub0llmregistry", "config.json")


def defaultRoot():
    """$SUB0_MODEL_REGISTRY, else "root" in the user config (~/.config/sub0llmregistry/config.json, %APPDATA% on Windows),
    else D:/LlmModels (/mnt/d/LlmModels under WSL)."""
    r = os.environ.get("SUB0_MODEL_REGISTRY", "")
    if not r:
        try:
            with open(userConfigPath(), encoding="utf-8-sig") as f:
                r = json.load(f).get("root") or ""
        except (OSError, ValueError):
            pass
    if not r:
        r = "D:/LlmModels" if os.name == "nt" else "/mnt/d/LlmModels"
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", r):  # a Windows path seen from WSL
        r = "/mnt/" + r[0].lower() + "/" + r[3:].replace("\\", "/")
    return r


def roleOf(name):
    n = name.lower()
    if "mmproj" in n:
        return "mmproj"
    m = SPLIT.search(n)
    if m:
        return "model" if m.group(1) == "00001" else "shard"
    if "imatrix" in n:
        return "imatrix"
    if n.endswith((".gguf", ".safetensors", ".bin", ".onnx", ".pt", ".pth")):
        return "model"
    if n.endswith(("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt")):
        return "tokenizer"
    if n.endswith(".json"):
        return "config"
    return "other"


class Evicted(LookupError):
    """The version's files were evicted from this machine (its manifest remains); restore them to use it."""

    def __init__(self, rid):
        LookupError.__init__(self, "%s was evicted from this machine; restore it with: python -m sub0llmregistry restore %s"
                             % (rid, rid))
        self.id = rid


def filePath(reg, m, name):
    return os.path.join(reg.root, *m["path"].split("/"), *re.split(r"[\\/]", name))


class Registry:
    def __init__(self, root=None):
        self.root = os.path.abspath(root or defaultRoot())

    # ---- reading ---------------------------------------------------------------------------------------------------------------
    def idFor(self, folder):
        rel = os.path.relpath(folder, self.root).replace("\\", "/")
        parts = rel.split("/")
        if len(parts) < 3 or rel.startswith(".."):
            raise ValueError("not a version folder (needs <area>/.../<version> under %s): %s" % (self.root, folder))
        return "/".join(parts[:-1]) + "@" + parts[-1]

    def folderFor(self, rid):
        base = rid.partition("#")[0]
        path, _, version = base.rpartition("@")
        return os.path.join(self.root, *path.split("/"), version)

    def manifests(self):
        """Every version's manifest (with "path": the folder relative to the root), sorted by id."""
        out = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            if MANIFEST in filenames:
                with open(os.path.join(dirpath, MANIFEST), encoding="utf-8-sig") as f:
                    m = json.load(f)
                m["path"] = os.path.relpath(dirpath, self.root).replace("\\", "/")
                out.append(m)
                dirnames[:] = []  # a version folder holds no further versions
        return sorted(out, key=lambda m: m["id"])

    def aliases(self):
        p = os.path.join(self.root, "aliases.json")
        if not os.path.exists(p):
            return {}
        with open(p, encoding="utf-8-sig") as f:
            return {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    def get(self, rid):
        folder = self.folderFor(rid)
        p = os.path.join(folder, MANIFEST)
        if not os.path.exists(p):
            raise KeyError("no registry entry: %s" % rid)
        with open(p, encoding="utf-8-sig") as f:
            m = json.load(f)
        m["path"] = os.path.relpath(folder, self.root).replace("\\", "/")
        return m

    def resolve(self, ref, restore=None, track=True, say=print):
        """An alias, id or id#file -> (absolute file or folder path, full id). See spec 'Ids' for the main-file rule.

        Records the use in usage.json (`track`). An evicted version raises Evicted unless `restore` (default: config.json
        cache.autoRestore) is true, in which case its files are fetched again first."""
        from . import cache
        al = self.aliases()
        rid = al.get(ref, ref)
        base, _, fname = rid.partition("#")
        m = self.get(base)
        if m.get("evicted"):
            if not (restore if restore is not None else cache.config(self).get("autoRestore")):
                raise Evicted(base)
            m = cache.restore(self, base, yes=True, track=False, say=say)
        if track:
            cache.touch(self, base)
        folder = os.path.join(self.root, *m["path"].split("/"))
        if not fname:
            models = [f["name"] for f in m["files"] if (f.get("role") or roleOf(f["name"])) == "model"]
            if len(models) == 1:
                fname = models[0]
            elif len(models) > 1:
                raise ValueError("%s holds %d model files; name one with #file: %s" % (base, len(models), ", ".join(models)))
        if fname:
            return os.path.join(folder, *re.split(r"[\\/]", fname)), base + "#" + fname
        return folder, base

    def query(self, **want):
        """Manifests matching every filter: format, area, architecture, text (in id/name/notes), maxBytes, minContext, moe."""
        out = []
        for m in self.manifests():
            f = m.get("facts") or {}
            size = sum(x.get("bytes", 0) for x in m["files"] if (x.get("role") or roleOf(x["name"])) in ("model", "shard"))
            if want.get("format") and m.get("format") != want["format"]:
                continue
            if want.get("area") and not m["id"].startswith(want["area"] + "/"):
                continue
            if want.get("architecture") and f.get("architecture") != want["architecture"]:
                continue
            if want.get("text") and want["text"].lower() not in (m["id"] + " " + str(f.get("name", "")) + " " + m.get("notes", "")).lower():
                continue
            if want.get("maxBytes") and size > want["maxBytes"]:
                continue
            if want.get("minContext") and (f.get("contextTrain") or 0) < want["minContext"]:
                continue
            if want.get("moe") is not None and bool(f.get("expertCount")) != want["moe"]:
                continue
            out.append(m)
        return out

    # ---- writing ---------------------------------------------------------------------------------------------------------------
    def writeManifest(self, folder, m):
        m = {k: v for k, v in m.items() if k != "path"}
        tmp = os.path.join(folder, MANIFEST + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(m, f, indent=2)
            f.write("\n")
        os.replace(tmp, os.path.join(folder, MANIFEST))

    def newManifest(self, folder, fmt="unknown", source=None, derivedFrom=None, tool=None, notes=""):
        folder = os.path.abspath(folder)
        if os.path.exists(os.path.join(folder, MANIFEST)):
            raise FileExistsError("already has a manifest (versions are immutable; make a new version): %s" % folder)
        files = []
        for dirpath, _, names in os.walk(folder):
            for n in names:
                if n == MANIFEST or n.endswith((".log", ".tmp")):
                    continue
                full = os.path.join(dirpath, n)
                rel = os.path.relpath(full, folder).replace("\\", "/")
                files.append({"name": rel, "bytes": os.path.getsize(full), "sha256": None, "role": roleOf(rel)})
        files.sort(key=lambda f: f["name"])
        m = {"schema": SCHEMA, "id": self.idFor(folder), "format": fmt, "files": files, "source": source, "derivedFrom": derivedFrom,
             "tool": tool, "created": datetime.date.today().isoformat(), "notes": notes}
        self.writeManifest(folder, m)
        return m

    def setAlias(self, name, rid):
        al = self.aliases()
        if rid:
            self.get(rid.partition("#")[0])
            al[name] = rid
        else:
            al.pop(name, None)
        with open(os.path.join(self.root, "aliases.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(dict(sorted(al.items())), f, indent=2)
            f.write("\n")

    def index(self):
        """Rewrite registry.json from the manifests and aliases. Returns (entries, aliases, warnings)."""
        warn = []
        entries = {}
        for m in self.manifests():
            expected = self.idFor(os.path.join(self.root, *m["path"].split("/")))
            if m["id"] != expected:
                warn.append("manifest id %s does not match its folder (%s)" % (m["id"], expected))
            rel = m["path"]  # entries carry "dir" (relative, for readers on another mount) and "path" (absolute, as generated)
            entries[m["id"]] = dict(m, dir=rel, path=os.path.join(self.root, *rel.split("/")))
        al = self.aliases()
        for k, v in al.items():
            if v.partition("#")[0] not in entries:
                warn.append("alias %s points at a missing entry: %s" % (k, v))
        out = {"format": "sub0llm-model-registry", "version": 1, "generated": datetime.datetime.now().isoformat(timespec="seconds"),
               "root": self.root, "aliases": al, "entries": entries}
        with open(os.path.join(self.root, "registry.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
        readme = os.path.join(self.root, "README.md")
        if not os.path.exists(readme):
            with open(readme, "w", encoding="utf-8", newline="\n") as f:
                f.write("# Model registry\n\nThis folder is a Sub0LlmModelRegistry (format in FORMAT.json): "
                        "https://github.com/CraigHutchinson/Sub0LlmModelRegistry\n\nRefer to models by alias or id, never by path:\n\n"
                        "    python -m sub0llmregistry list\n    python -m sub0llmregistry resolve <alias>\n\n"
                        "Versions marked evicted keep their manifest; `restore` fetches their files again.\n")
        marker = os.path.join(self.root, "FORMAT.json")
        if not os.path.exists(marker):
            with open(marker, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"format": "sub0llm-model-registry", "version": 1}, f)
                f.write("\n")
        return entries, al, warn

    def verify(self, hashes=False, say=print):
        """Every manifest's files exist with the recorded size; with `hashes`, sha256 is checked, or recorded where missing."""
        problems = 0
        for m in self.manifests():
            folder = os.path.join(self.root, *m["path"].split("/"))
            if m.get("evicted"):
                continue  # files deliberately absent; restore checks every sha256
            changed = False
            for f in m["files"]:
                p = os.path.join(folder, *re.split(r"[\\/]", f["name"]))
                if not os.path.exists(p):
                    say("%s: missing %s" % (m["id"], f["name"]))
                    problems += 1
                    continue
                if os.path.getsize(p) != f["bytes"]:
                    say("%s: %s is %d bytes, manifest says %d" % (m["id"], f["name"], os.path.getsize(p), f["bytes"]))
                    problems += 1
                if hashes:
                    h = sha256(p)
                    if f.get("sha256") and f["sha256"] != h:
                        say("%s: %s sha256 mismatch" % (m["id"], f["name"]))
                        problems += 1
                    elif not f.get("sha256"):
                        f["sha256"] = h
                        changed = True
            if changed:
                self.writeManifest(folder, m)
                say("%s: sha256 recorded" % m["id"])
        return problems

    def scanFacts(self, rid=None, force=False, say=print):
        """Read GGUF headers into each manifest's "facts" (only the header is read)."""
        from . import gguf
        n = 0
        for m in ([self.get(rid)] if rid else self.manifests()):
            if (m.get("facts") and not force) or m.get("evicted"):
                continue
            folder = os.path.join(self.root, *m["path"].split("/"))
            names = [f["name"] for f in m["files"] if f["name"].lower().endswith(".gguf") and (f.get("role") or roleOf(f["name"])) in ("model", "shard")]
            groups = {}
            for nme in names:  # one model per main file: shards group with their first shard
                key = SPLIT.sub("", nme)
                groups.setdefault(key, []).append(nme)
            if not groups:
                continue
            facts = {}
            for key, members in sorted(groups.items()):
                paths = [os.path.join(folder, *re.split(r"[\\/]", x)) for x in sorted(members)]
                try:
                    facts[sorted(members)[0]] = gguf.facts(paths)
                except Exception as e:
                    say("%s: %s: %s" % (m["id"], key, e))
            if facts:
                m["facts"] = next(iter(facts.values())) if len(facts) == 1 else {"byFile": facts}
                self.writeManifest(folder, m)
                say("%s: facts recorded" % m["id"])
                n += 1
        return n


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with openNoAtime(path) as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def openNoAtime(path):
    """Open for reading without updating the access time where the OS allows it (hashing is not a use of the model)."""
    flag = getattr(os, "O_NOATIME", 0)
    if flag:
        try:
            return os.fdopen(os.open(path, os.O_RDONLY | flag), "rb")
        except OSError:
            pass
    return open(path, "rb")
