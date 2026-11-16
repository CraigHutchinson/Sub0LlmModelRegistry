"""Hugging Face: hf:<org>/<repo>[@<revision>][#<file or glob>]

The revision (branch, tag or commit; default main) is pinned to its commit: the version folder is hf/<org>/<repo>/<commit[:12]>.
Sizes and sha256 come from the Hub's LFS metadata. A gated or private repo needs HF_TOKEN in the environment.
"""
import fnmatch
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from . import Offer, Provider, request

API = "https://huggingface.co"
SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


class HuggingFace(Provider):
    name = "hf"
    description = "Hugging Face Hub: hf:org/repo[@revision][#file-or-glob] (HF_TOKEN for gated repos)"

    def headers(self):
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        return {"Authorization": "Bearer " + tok} if tok else {}

    def info(self, repo, revision):
        url = "%s/api/models/%s/revision/%s?blobs=true" % (API, repo, urllib.parse.quote(revision, safe=""))
        try:
            with urllib.request.urlopen(request(url, self.headers())) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise SystemExit("hf: %s@%s: HTTP %d%s" % (repo, revision, e.code, " (gated or private: set HF_TOKEN)" if e.code in (401, 403) else ""))

    def offer(self, spec):
        m = re.match(r"^([^/@#\s]+/[^/@#\s]+)(?:@([^#\s]+))?(?:#(.+))?$", spec)
        if not m:
            raise SystemExit("hf: expected org/repo[@revision][#file-or-glob], got %r" % spec)
        repo, revision, pattern = m.group(1), m.group(2) or "main", m.group(3)
        info = self.info(repo, revision)
        commit = info.get("sha") or revision
        sib = info.get("siblings", [])
        names = [s["rfilename"] for s in sib]
        if pattern:
            chosen = [n for n in names if fnmatch.fnmatch(n, pattern) or n == pattern]
            # one shard named: take the whole split
            for n in list(chosen):
                g = SPLIT.search(n)
                if g:
                    stem = n[: g.start()]
                    chosen += [x for x in names if x.startswith(stem) and SPLIT.search(x) and x not in chosen]
        else:
            ggufs = [n for n in names if n.lower().endswith(".gguf")]
            mains = sorted({SPLIT.sub("", n) for n in ggufs if "mmproj" not in n.lower()})
            if len(mains) > 1:
                raise SystemExit("hf: %s has %d models; pick one with #file or #glob:\n  %s" % (repo, len(mains), "\n  ".join(sorted(ggufs))))
            chosen = ggufs or [n for n in names if n.endswith((".safetensors", ".json", ".model", ".txt"))]
        if not chosen:
            raise SystemExit("hf: nothing in %s@%s matches %r" % (repo, revision, pattern))
        files = []
        for s in sib:
            if s["rfilename"] not in chosen:
                continue
            lfs = s.get("lfs") or {}
            files.append({"name": s["rfilename"], "bytes": lfs.get("size") or s.get("size"), "sha256": lfs.get("sha256"),
                          "url": "%s/%s/resolve/%s/%s" % (API, repo, commit, urllib.parse.quote(s["rfilename"]))})
        fmt = "gguf" if any(f["name"].lower().endswith(".gguf") for f in files) else "safetensors" if any(f["name"].endswith(".safetensors") for f in files) else "other"
        o = Offer("hf/%s/%s" % (repo, commit[:12]), sorted(files, key=lambda f: f["name"]),
                  {"provider": "hf", "repo": repo, "revision": commit, "url": "%s/%s" % (API, repo)}, fmt=fmt)
        o.headers = self.headers()
        o.pinned = bool(info.get("sha"))
        return o
