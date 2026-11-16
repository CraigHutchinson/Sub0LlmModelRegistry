import json
import os
import struct
import tempfile
import unittest

from sub0llmregistry import Registry, gguf


def ggufBytes(blocks=2, experts=True):
    """A tiny GGUF header: metadata plus a tensor table (no weights needed by the reader)."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [("general.architecture", 8, s("toy")), ("toy.block_count", 4, struct.pack("<I", blocks)),
          ("toy.attention.head_count_kv", 4, struct.pack("<I", 2)), ("toy.attention.key_length", 4, struct.pack("<I", 64)),
          ("toy.expert_count", 4, struct.pack("<I", 4))]
    tensors = [("blk.%d.attn_k.weight" % i, [64, 128], 1) for i in range(blocks)]
    if experts:
        tensors += [("blk.%d.ffn_up_exps.weight" % i, [64, 128, 4], 0) for i in range(blocks)]
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, t, v in kv:
        out += s(k) + struct.pack("<I", t) + v
    off = 0
    for name, dims, t in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", t, off)
        n = 1
        for d in dims:
            n *= d
        off += n * (2 if t == 1 else 4)
    return out


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.folder = os.path.join(self.root, "local", "toy", "f16", "20261116")
        os.makedirs(self.folder)
        with open(os.path.join(self.folder, "toy.gguf"), "wb") as f:
            f.write(ggufBytes())
        with open(os.path.join(self.folder, "mmproj-toy.gguf"), "wb") as f:
            f.write(ggufBytes(experts=False))
        self.reg = Registry(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_resolve_alias_index(self):
        m = self.reg.newManifest(self.folder, fmt="gguf")
        self.assertEqual(m["id"], "local/toy/f16@20261116")
        self.assertEqual({f["name"]: f["role"] for f in m["files"]}, {"toy.gguf": "model", "mmproj-toy.gguf": "mmproj"})
        path, rid = self.reg.resolve("local/toy/f16@20261116")
        self.assertTrue(path.endswith("toy.gguf") and "mmproj" not in path)
        self.reg.setAlias("toy", "local/toy/f16@20261116")
        self.assertEqual(self.reg.resolve("toy")[0], path)
        entries, al, warn = self.reg.index()
        self.assertEqual(warn, [])
        idx = json.load(open(os.path.join(self.root, "registry.json")))
        e = idx["entries"]["local/toy/f16@20261116"]
        self.assertEqual(e["dir"], "local/toy/f16/20261116")
        self.assertTrue(os.path.isabs(e["path"]))
        self.assertTrue(os.path.exists(os.path.join(self.root, "FORMAT.json")))
        with self.assertRaises(FileExistsError):
            self.reg.newManifest(self.folder)

    def test_facts_and_query(self):
        self.reg.newManifest(self.folder, fmt="gguf")
        self.assertEqual(self.reg.scanFacts(say=lambda *_: None), 1)
        f = self.reg.get("local/toy/f16@20261116")["facts"]
        self.assertEqual((f["blockCount"], f["kvLayers"], f["expertLayers"]), (2, 2, 2))
        self.assertEqual(f["kvBytesPerTokenF16"], 2 * 2 * (64 + 64) * 2)
        self.assertEqual(f["expertBytesByLayer"], [64 * 128 * 4 * 4] * 2)
        self.assertEqual(len(self.reg.query(moe=True)), 1)
        self.assertEqual(len(self.reg.query(architecture="other")), 0)

    def test_verify(self):
        self.reg.newManifest(self.folder, fmt="gguf")
        self.assertEqual(self.reg.verify(hashes=True, say=lambda *_: None), 0)
        self.assertTrue(all(x["sha256"] for x in self.reg.get("local/toy/f16@20261116")["files"]))
        with open(os.path.join(self.folder, "toy.gguf"), "ab") as fh:
            fh.write(b"x")
        self.assertEqual(self.reg.verify(say=lambda *_: None), 1)

    def test_gguf_reader(self):
        r = gguf.read(os.path.join(self.folder, "toy.gguf"))
        self.assertEqual(r["metadata"]["general.architecture"], "toy")
        self.assertEqual(len(r["tensors"]), 4)


if __name__ == "__main__":
    unittest.main()
