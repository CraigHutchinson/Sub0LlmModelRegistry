"""Any direct download: url:https://host/path/file.gguf[#sha256=<hex>]

The version folder is url/<host>/<file stem>/<YYYYMMDD> (the day it was fetched). Give #sha256= when the publisher lists one.
"""
import datetime
import os
import re
import urllib.parse
import urllib.request

from . import Offer, Provider, request


class Url(Provider):
    name = "url"
    description = "a direct https download: url:https://host/path/file[#sha256=hex]"

    def offer(self, spec):
        link, _, frag = spec.partition("#")
        u = urllib.parse.urlparse(link)
        if u.scheme != "https":
            raise SystemExit("url: only https downloads are accepted: %r" % link)
        sha = re.match(r"^sha256=([0-9a-fA-F]{64})$", frag)
        name = os.path.basename(urllib.parse.unquote(u.path)) or "download"
        size = None
        try:
            with urllib.request.urlopen(request(link, method="HEAD")) as r:
                size = int(r.headers.get("Content-Length") or 0) or None
        except Exception:
            pass
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", os.path.splitext(name)[0]).strip("-").lower() or "download"
        fmt = os.path.splitext(name)[1].lstrip(".").lower() or "other"
        return Offer("url/%s/%s/%s" % (u.hostname, stem, datetime.date.today().strftime("%Y%m%d")),
                     [{"name": name, "url": link, "bytes": size, "sha256": sha.group(1).lower() if sha else None}],
                     {"provider": "url", "repo": None, "revision": None, "url": link}, fmt=fmt)
