"""GGUF header reader (metadata and tensor table only; the weights are never read) and the facts derived from it.

Facts are what an engine needs to plan placement before measuring: layer and expert counts, bytes per expert layer, the
size of everything else, and KV cache bytes per token. For a split GGUF, pass every shard; tensors are summed over them.
"""
import re
import struct

# ggml tensor types: (block size in elements, bytes per block). Enough to size every tensor in common GGUFs.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17),
}

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
ARRAY_KEEP = 4096  # arrays longer than this (token lists) are skipped and recorded by length only


class _Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        n = struct.calcsize(fmt)
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError("truncated GGUF header")
        return struct.unpack(fmt, data)[0]

    def string(self):
        n = self.take("<Q")
        return self.f.read(n).decode("utf-8", "replace")

    def skipString(self):
        self.f.seek(self.take("<Q"), 1)

    def value(self, t):
        if t in _SCALARS:
            return self.take(_SCALARS[t])
        if t == 8:
            return self.string()
        if t == 9:
            it, n = self.take("<I"), self.take("<Q")
            if n > ARRAY_KEEP:
                if it in _SCALARS:
                    self.f.seek(n * struct.calcsize(_SCALARS[it]), 1)
                elif it == 8:
                    for _ in range(n):
                        self.skipString()
                else:
                    for _ in range(n):
                        self.value(it)
                return {"_array": True, "type": it, "length": n}
            return [self.value(it) for _ in range(n)]
        raise ValueError("unknown GGUF value type %d" % t)


def read(path):
    """{"version", "metadata": {key: value}, "tensors": [{"name", "shape", "type", "bytes"}]} of one GGUF file."""
    from .store import openNoAtime
    with openNoAtime(path) as f:
        r = _Reader(f)
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file: %s" % path)
        version = r.take("<I")
        if version < 2:
            raise ValueError("GGUF version %d is not supported" % version)
        nTensors, nKv = r.take("<Q"), r.take("<Q")
        meta = {}
        for _ in range(nKv):
            key = r.string()
            meta[key] = r.value(r.take("<I"))
        infos = []
        for _ in range(nTensors):
            name = r.string()
            dims = [r.take("<Q") for _ in range(r.take("<I"))]
            ttype = r.take("<I")
            offset = r.take("<Q")
            infos.append((name, dims, ttype, offset))
        align = int(meta.get("general.alignment", 32))
        dataStart = (f.tell() + align - 1) // align * align
        f.seek(0, 2)
        size = f.tell()
    tensors = []
    order = sorted(range(len(infos)), key=lambda i: infos[i][3])
    for pos, i in enumerate(order):
        name, dims, ttype, offset = infos[i]
        elems = 1
        for d in dims:
            elems *= d
        if ttype in GGML_TYPES:
            _, block, nbytes = GGML_TYPES[ttype]
            b = elems // block * nbytes
        else:  # unknown type: the gap to the next tensor
            nxt = infos[order[pos + 1]][3] if pos + 1 < len(order) else size - dataStart
            b = nxt - offset
        tensors.append({"name": name, "shape": dims, "type": GGML_TYPES.get(ttype, ("type%d" % ttype,))[0], "elements": elems, "bytes": b})
    return {"version": version, "metadata": meta, "tensors": tensors}


EXPERT = re.compile(r"^blk\.(\d+)\.ffn_.*_exps(\.|$)")
ATTN_K = re.compile(r"^blk\.(\d+)\.attn_(k|kv_a_mqa|k_b)\.weight$")
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
              13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS",
              21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
              29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE"}


def facts(paths):
    """Planning facts over one GGUF or all shards of a split one."""
    parts = [read(p) for p in paths]
    meta = parts[0]["metadata"]
    tensors = [t for p in parts for t in p["tensors"]]
    arch = meta.get("general.architecture", "")

    def m(key, default=None):
        v = meta.get(arch + "." + key, default)
        return v if not isinstance(v, dict) else default

    blocks = m("block_count")
    total = sum(t["bytes"] for t in tensors)
    perLayer = {}
    for t in tensors:
        g = EXPERT.match(t["name"])
        if g:
            perLayer[int(g.group(1))] = perLayer.get(int(g.group(1)), 0) + t["bytes"]
    expertBytes = sum(perLayer.values())
    kvLayers = sorted({int(g.group(1)) for t in tensors for g in [ATTN_K.match(t["name"])] if g})
    headsKv = m("attention.head_count_kv")
    if isinstance(headsKv, list):  # per-layer list: count only layers that keep a KV cache
        headsKvList = headsKv
        headsKv = max(headsKv) if headsKv else None
    else:
        headsKvList = None
    keyLen = m("attention.key_length") or (m("embedding_length") and m("attention.head_count") and m("embedding_length") // m("attention.head_count"))
    valLen = m("attention.value_length") or keyLen
    kvPerToken = None
    if kvLayers and headsKv and keyLen:
        if headsKvList:
            kvPerToken = sum(headsKvList[i] * (keyLen + valLen) for i in kvLayers if i < len(headsKvList)) * 2
        else:
            kvPerToken = len(kvLayers) * headsKv * (keyLen + valLen) * 2
    out = {
        "architecture": arch,
        "name": meta.get("general.name"),
        "sizeLabel": meta.get("general.size_label"),
        "fileType": FILE_TYPES.get(meta.get("general.file_type"), meta.get("general.file_type")),
        "parameters": sum(t["elements"] for t in tensors),
        "tensorBytes": total,
        "blockCount": blocks,
        "contextTrain": m("context_length"),
        "embedding": m("embedding_length"),
        "headCount": m("attention.head_count") if not isinstance(m("attention.head_count"), list) else max(m("attention.head_count")),
        "headCountKv": headsKv,
        "keyLength": keyLen,
        "valueLength": valLen,
        "slidingWindow": m("attention.sliding_window"),
        "fullAttentionInterval": m("full_attention_interval"),
        "kvLayers": len(kvLayers) or None,
        "kvBytesPerTokenF16": kvPerToken,
        "kvBytesPerTokenQ8": kvPerToken * 34 // 64 if kvPerToken else None,
        "expertCount": m("expert_count"),
        "expertUsed": m("expert_used_count"),
        "expertLayers": len(perLayer) or None,
        "expertBytes": expertBytes or None,
        "expertBytesPerLayerMax": max(perLayer.values()) if perLayer else None,
        "expertBytesPerLayerMean": expertBytes // len(perLayer) if perLayer else None,
        # llama.cpp's --n-cpu-moe N moves the experts of layers 0..N-1: the GPU keeps sum(expertBytesByLayer[N:])
        "expertBytesByLayer": [perLayer.get(i, 0) for i in range(blocks)] if perLayer and blocks else None,
        "nonExpertBytes": total - expertBytes,
        "shards": len(paths),
        "ggufVersion": parts[0]["version"],
    }
    return {k: v for k, v in out.items() if v is not None}
