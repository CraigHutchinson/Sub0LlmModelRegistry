# Sub0LlmModelRegistry

One folder of model weights for every tool on the machine: each model stored once, in an immutable versioned folder
with a manifest that records where it came from, its files, sizes, hashes, and facts read from the files themselves.
Engines (Sub0LlmEngine), campaigns (Sub0Squad) and transformations (Sub0Llm) refer to models by **alias** or **id**
and never by path.

* **The format**: [spec/LAYOUT.md](spec/LAYOUT.md) and [spec/manifest.schema.json](spec/manifest.schema.json). Plain
  folders and JSON, so any language can read it; `registry.json` is a generated index for readers that do no more.
* **The reference tool**: the `sub0llmregistry` Python package (standard library only), a CLI over the same.
* **Providers**: plugins that fetch models into the registry (Hugging Face and plain https built in; more by entry point).

```
D:\LlmModels\                     the data (this repository holds none of it)
  FORMAT.json  aliases.json  registry.json  MIGRATION.log
  hf/<org>/<repo>/<revision>/      downloads, as published
  sub0llm/<model>/<variant>/<version>/   Sub0Llm transformations (custom formats)
  local/<model>/<variant>/<version>/     our own derivations (quantizations, merges, adapters)
  _incoming/                       partial downloads
```

## Use

```bash
python -m sub0llmregistry list                      # everything, with size, architecture, quantization and aliases
python -m sub0llmregistry list qwen --moe --max-gib 20 --min-context 131072
python -m sub0llmregistry resolve qwen36-35b-a3b    # alias / id / id#file -> absolute path
python -m sub0llmregistry show qwen36-35b-a3b       # the manifest, facts included
python -m sub0llmregistry alias qwen36 hf/mudler/Qwen3.6-35B-A3B-APEX-GGUF@unpinned-20260520
python -m sub0llmregistry facts                     # read GGUF headers into manifests (header only: about a second for all)
python -m sub0llmregistry verify [--hash]           # files present with their recorded sizes (and hashes)
python -m sub0llmregistry index                     # regenerate registry.json after editing aliases.json by hand
python -m sub0llmregistry new-manifest D:/LlmModels/local/mymodel/q4/20261116 --format gguf
```

The root is `--root`, else `$SUB0_MODEL_REGISTRY`, else `D:/LlmModels` (`/mnt/d/LlmModels` under WSL; a Windows path in
the variable is translated).

From Python:

```python
from sub0llmregistry import Registry
reg = Registry()
path, rid = reg.resolve("qwen36-35b-a3b")
facts = reg.get(rid.partition("#")[0])["facts"]      # blockCount, expertBytesByLayer, kvBytesPerTokenQ8, ...
moe = reg.query(moe=True, maxBytes=20 * 2**30)
```

## Facts

`facts` reads the GGUF header and tensor table (never the weights) and records what an engine needs to plan placement
before measuring anything: layer count, KV layers and KV bytes per token (F16 and q8_0), expert count and experts used,
expert bytes per layer (with the per-layer list, so llama.cpp's `--n-cpu-moe N` VRAM is
`nonExpertBytes + sum(expertBytesByLayer[N:])` plus buffers), training context, architecture and quantization.

## Fetching models

```bash
python -m sub0llmregistry fetch hf:unsloth/Qwen3-8B-GGUF#Qwen3-8B-Q4_K_M.gguf --alias qwen3-8b
python -m sub0llmregistry fetch url:https://example.org/model.gguf#sha256=<hex>
```

`fetch` lists the files, sizes and target folder and asks before downloading anything (`--yes` when the owner has already
agreed). Files land in `_incoming/` (an interrupted fetch resumes), are checked against the provider's sizes and sha256,
and only then move into their version folder with a manifest; facts are read and the index regenerated. A Hugging Face
revision is pinned to its commit (`hf/<org>/<repo>/<commit[:12]>`). Gated repositories read `HF_TOKEN` from the
environment; the tool never asks for or stores credentials.

### Writing a provider

```python
from sub0llmregistry.providers import Offer, Provider

class MyHub(Provider):
    name = "myhub"
    description = "myhub:<model>[@<version>]"
    def offer(self, spec):          # read the provider's index only; download nothing
        return Offer("myhub/<model>/<version>", [{"name": "m.gguf", "url": "https://...", "bytes": 123, "sha256": "..."}],
                     {"provider": "myhub", "repo": "<model>", "revision": "<version>", "url": "https://..."})
```

Register it from its own package:

```toml
[project.entry-points."sub0llmregistry.providers"]
myhub = "mypackage.myhub:MyHub"
```

## Status

Format version 1. The registry on the development machine holds 14 versions (5 GGUF downloads, 9 Sub0Llm
transformations). Next: a query service for engines on other hosts, more providers, and adapters/derivations recorded
with `derivedFrom` and `tool` when Sub0Llm writes a new version.

MIT licensed.
