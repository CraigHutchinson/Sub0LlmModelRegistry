import datetime
import hashlib
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from sub0llmregistry import Evicted, Registry, cache, providers

QUIET = lambda *_: None  # noqa: E731


class Fake(providers.Provider):
    """Serves files from a local folder as file:// URLs: fake:<name>  ->  fake/<name>/<rev>."""
    name = "fake"
    description = "test"

    def __init__(self, upstream):
        self.upstream = upstream

    def offer(self, spec):
        p = os.path.join(self.upstream, spec)
        with open(p, "rb") as fh:
            data = fh.read()
        o = providers.Offer("fake/%s/rev1" % spec.replace(".", "-"),
                            [{"name": spec, "url": pathlib.Path(p).as_uri(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
                            {"provider": "fake", "repo": spec, "revision": "rev1", "url": None}, fmt="other")
        o.pinned = True
        return o

    def refetch(self, m):
        return [dict(f, url=pathlib.Path(os.path.join(self.upstream, f["name"])).as_uri()) for f in m["files"]]


class Cache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "reg")
        self.up = os.path.join(self.tmp.name, "upstream")
        os.makedirs(self.root)
        os.makedirs(self.up)
        self.reg = Registry(self.root)
        self.fake = Fake(self.up)
        p = mock.patch.object(providers, "available", lambda: {"fake": self.fake})
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, name, size, daysAgo):
        with open(os.path.join(self.up, name), "wb") as f:
            f.write(os.urandom(size))
        m = providers.fetch(self.reg, self.fake.offer(name), say=QUIET)
        # pretend it was created, and last used, `daysAgo` days ago
        m["created"] = (datetime.date.today() - datetime.timedelta(days=daysAgo)).isoformat()
        self.reg.writeManifest(self.reg.folderFor(m["id"]), m)
        t = (datetime.datetime.now() - datetime.timedelta(days=daysAgo)).timestamp()
        for f in m["files"]:
            os.utime(os.path.join(self.reg.folderFor(m["id"]), f["name"]), (t, t))
        return m["id"]

    def test_lru_plan_respects_limit_idle_and_keep(self):
        old = self.add("old.bin", 1000, 40)
        mid = self.add("mid.bin", 1000, 20)
        new = self.add("new.bin", 1000, 1)
        ids = lambda c: [x[0]["id"] for x in c]  # noqa: E731
        self.assertEqual(ids(cache.plan(self.reg, maxBytes=1500, idleDays=0)), [old, mid])
        self.assertEqual(ids(cache.plan(self.reg, maxBytes=2500, idleDays=0)), [old])
        self.assertEqual(ids(cache.plan(self.reg, maxBytes=0, idleDays=14)), [old, mid])  # `new` is too recent
        self.assertEqual(ids(cache.plan(self.reg, idleDays=30)), [old])  # no limit: everything idle that long
        cfg = dict(cache.DEFAULTS, keep=["o"])
        self.reg.setAlias("o", old)
        self.assertEqual(ids(cache.plan(self.reg, maxBytes=0, idleDays=0, cfg=cfg)), [mid, new])
        cache.touch(self.reg, old)  # a use moves it to the back of the queue
        self.assertEqual(ids(cache.plan(self.reg, maxBytes=2500, idleDays=0)), [mid])

    def test_evict_resolve_restore(self):
        rid = self.add("m.bin", 4096, 30)
        path, _ = self.reg.resolve(rid)
        cache.evict(self.reg, self.reg.get(rid), say=QUIET)
        self.assertFalse(os.path.exists(path))
        m = self.reg.get(rid)
        self.assertEqual(m["evicted"]["bytes"], 4096)
        self.assertTrue(m["files"][0]["sha256"])
        self.assertEqual(self.reg.verify(say=QUIET), 0)  # evicted files are not problems
        with self.assertRaises(Evicted):
            self.reg.resolve(rid)
        self.assertEqual(self.reg.resolve(rid, restore=True, say=QUIET)[0], path)
        self.assertTrue(os.path.exists(path))
        self.assertNotIn("evicted", self.reg.get(rid))
        self.assertEqual(self.reg.verify(hashes=True, say=QUIET), 0)
        self.assertEqual(cache.usage(self.reg)[rid]["uses"], 2)

    def test_restore_rejects_changed_upstream(self):
        rid = self.add("m.bin", 4096, 30)
        cache.evict(self.reg, self.reg.get(rid), say=QUIET)
        with open(os.path.join(self.up, "m.bin"), "wb") as f:
            f.write(os.urandom(4096))
        with self.assertRaises(IOError):
            cache.restore(self.reg, rid, yes=True, say=QUIET)
        self.assertTrue(self.reg.get(rid)["evicted"])

    def test_auto_restore_from_config(self):
        rid = self.add("m.bin", 100, 30)
        cache.evict(self.reg, self.reg.get(rid), say=QUIET)
        with open(os.path.join(self.root, "config.json"), "w") as f:
            json.dump({"cache": {"autoRestore": True}}, f)
        self.assertTrue(os.path.exists(self.reg.resolve(rid, say=QUIET)[0]))

    def test_never_evicts_what_we_made(self):
        folder = os.path.join(self.root, "local", "toy", "q4", "20261116")
        os.makedirs(folder)
        open(os.path.join(folder, "toy.gguf"), "wb").write(b"x")
        m = self.reg.newManifest(folder)
        self.assertIn("made here", cache.whyNot(self.reg, m))
        self.assertEqual(cache.plan(self.reg, maxBytes=0, idleDays=0), [])

    def test_adopt_moves_links_and_logs(self):
        with open(os.path.join(self.up, "a.bin"), "wb") as f:
            f.write(b"abc" * 100)
        old = os.path.join(self.tmp.name, "Old-Folder")
        os.makedirs(old)
        with open(os.path.join(old, "a.bin"), "wb") as f:
            f.write(b"abc" * 100)
        m = providers.adopt(self.reg, old, "fake:a.bin", alias="a", say=QUIET)
        self.assertEqual(m["id"], "fake/a-bin@rev1")
        self.assertTrue(os.path.islink(old))
        self.assertTrue(os.path.exists(os.path.join(old, "a.bin")))  # the old path still works
        self.assertEqual(self.reg.resolve("a")[0], os.path.join(self.reg.folderFor(m["id"]), "a.bin"))
        self.assertIn("fake/a-bin@rev1", open(os.path.join(self.root, "MIGRATION.log")).read())

    def test_adopt_refuses_different_bytes(self):
        with open(os.path.join(self.up, "a.bin"), "wb") as f:
            f.write(b"abc" * 100)
        old = os.path.join(self.tmp.name, "Old")
        os.makedirs(old)
        with open(os.path.join(old, "a.bin"), "wb") as f:
            f.write(b"abd" * 100)
        with self.assertRaises(SystemExit):
            providers.adopt(self.reg, old, "fake:a.bin", say=QUIET)
        self.assertTrue(os.path.isfile(os.path.join(old, "a.bin")))


if __name__ == "__main__":
    unittest.main()
