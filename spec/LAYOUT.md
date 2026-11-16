# Registry format, version 1

A registry is a folder (for example `D:\LlmModels`, `/mnt/d/LlmModels` from WSL) that holds model weights, once each,
in versioned folders described by manifests. This document is the format; any language may read or write it. The
Python package in this repository is the reference implementation.

## Folder layout

```
<root>/
  FORMAT.json        {"format": "sub0llm-model-registry", "version": 1}  marks the folder as a registry
  README.md            a pointer to this project for whoever opens the folder
  aliases.json         hand-edited: short name -> registry id (moving an alias is how a model is upgraded)
  registry.json        GENERATED index of every manifest and alias; never edited, safe to delete and rebuild
  MIGRATION.log        one line per folder moved in from elsewhere
  config.json          optional, hand-edited: this machine's disk policy (see "Disk use")
  usage.json           machine state: when each version was last used; safe to delete
  hf/<org>/<repo>/<revision>/...              downloads from Hugging Face, exactly as published
  <provider>/<namespace...>/<revision>/...    downloads from another provider (its plugin names the folders)
  sub0llm/<model>/<variant>/<version>/...     Sub0Llm transformations (custom formats)
  local/<model>/<variant>/<version>/...       our own derivations (llama-quantize outputs, merges, adapters, imatrix files)
  _incoming/           partial downloads and work in progress; nothing may refer to a path in here
```

The first path segment is the **area**. Downloads go under their provider's name; anything we made goes under
`sub0llm/` or `local/`.

## Versions

Every leaf folder (`<revision>` or `<version>`) is one **version** and holds `MANIFEST.json`. Once written, a version's
**files never change**: a new model, quantization or transformation is a new version beside the old one. Manifest fields
that describe the files without changing them (`sha256`, `facts`, `notes`, `evicted`) may be filled in later.

| Part | Rule | Examples |
|---|---|---|
| `<org>/<repo>` | the provider's repository id, unchanged | `mudler/Qwen3.6-35B-A3B-APEX-GGUF` |
| `<revision>` | the provider's commit or version, first 12 characters of a hash; `unpinned-YYYYMMDD` (date obtained) when it was not recorded | `198f08841147`, `unpinned-20260520` |
| `<model>` | lower-case base model name, `-` separated | `qwen4-full48` |
| `<variant>` | what differs between siblings: precision, format, recipe | `bf16`, `fp8-q`, `o5-headonly` |
| `<version>` | `YYYYMMDD`, plus `-label` when two are made the same day | `20260923`, `20260911-recheck` |

## Ids

A version's id is its path from the root with `@` before the last segment; `#<file>` names one file in it:

```
hf/mudler/Qwen3.6-35B-A3B-APEX-GGUF@unpinned-20260520
hf/mudler/Qwen3.6-35B-A3B-APEX-GGUF@unpinned-20260520#Qwen3.6-35B-A3B-APEX-I-Compact.gguf
sub0llm/qwen4-full48/bf16@20260923
```

Resolving an id without `#file` gives the version's main file: its only model file, or the first shard of a split GGUF
(`-00001-of-0000N.gguf`; llama.cpp loads the others from the same folder), never an `mmproj` projector. A version with
several model files must be named with `#file`.

## MANIFEST.json

Schema: [manifest.schema.json](manifest.schema.json). In short:

```json
{
  "schema": "sub0llm-model-registry/manifest/1",
  "id": "hf/mudler/Qwen3.6-35B-A3B-APEX-GGUF@unpinned-20260520",
  "format": "gguf",
  "files": [{"name": "Qwen3.6-35B-A3B-APEX-I-Compact.gguf", "bytes": 17293089472, "sha256": null, "role": "model"}],
  "source": {"provider": "hf", "repo": "mudler/Qwen3.6-35B-A3B-APEX-GGUF", "revision": null, "url": null},
  "derivedFrom": null,
  "tool": null,
  "created": "2026-05-20",
  "facts": {"architecture": "qwen35moe", "blockCount": 40, "expertCount": 256, "...": "..."},
  "notes": ""
}
```

* `files[].name` is relative to the version folder (it may include sub-folders); `role` is `model`, `mmproj`,
  `shard`, `imatrix`, `adapter`, `tokenizer`, `config` or `other`.
* `source` is set for downloads; `derivedFrom` (an id) and `tool` (`{name, commit, command}`) for what we made.
* `evicted` is absent (or null) while the files are on this machine. When they have been deleted to save space it is
  `{"at": "<ISO time>", "bytes": <freed>}`: the folder and manifest stay, so the id, aliases, facts and hashes still
  answer questions, but the files must be restored before use (see "Disk use").
* `facts` is read from the files (GGUF headers) by the reference tool; consumers such as Sub0LlmEngine use it to
  plan placement before measuring (bytes per expert layer, KV bytes per token, layer counts).

## registry.json

Generated: `{"format", "version", "generated", "root", "aliases": {...}, "entries": {id: manifest + "dir" + "path"}}`,
where `dir` is the version folder relative to the root with `/` separators and `path` the same folder as an absolute path
on the machine that generated the file (`root` joined with `dir`). A reader that sees the registry under another mount
(WSL's `/mnt/d/...` for `D:\...`) joins its own root with `dir`. Readers that cannot run the reference tool
read this file; writers must regenerate it after adding a version or editing `aliases.json`.

## Disk use

A registry may hold more than the drive should keep. Versions that can be fetched again are **evicted** when unused
(their files deleted, `evicted` set in the manifest) and **restored** when next asked for.

* **Evictable** means: a download area whose provider can fetch the manifest's files again (`source`), with a `sha256`
  recorded for every file, not listed in `config.json` `cache.keep`. Versions under `local/` and `sub0llm/` are never
  evicted: nothing else holds a copy.
* **Restoring** downloads every file again from `source` (its `revision`, or the provider's default branch when the
  revision was not recorded), accepts it only if each `sha256` matches, and clears `evicted`. A version that no longer
  matches upstream stays evicted; the files never change.
* **Last use** is the latest of `usage.json` (written whenever the reference tool resolves a version), the files' access
  times (for readers that open paths directly; coarse under `relatime`) and the manifest's `created`.
* **Readers** of `registry.json` must check `evicted` before opening `path`.

`config.json` (all optional):

```json
{"cache": {"maxBytes": 214748364800, "minFreeBytes": 53687091200, "minIdleDays": 14,
           "keep": ["qwen36-35b-a3b"], "autoEvict": false, "autoRestore": false}}
```

| Key | Meaning |
|---|---|
| `maxBytes` | evict least recently used versions until the registry is no larger than this |
| `minFreeBytes` | evict until the drive has at least this much free |
| `minIdleDays` | never evict a version used more recently than this, whatever the limits say (default 14) |
| `keep` | aliases or ids never evicted |
| `autoEvict` | before a fetch or restore, evict by the policy above to make room |
| `autoRestore` | resolving an evicted version fetches it again without asking (otherwise it is an error naming the version) |

`usage.json`: `{id: {"lastUsed": "<ISO time>", "uses": n}}`. Losing it only makes last-use estimates coarser.

## Moving folders in

Move a folder into its version folder (same drive: a rename), leave a directory junction or symbolic link at the old
path so existing code keeps working, and append a line to `MIGRATION.log`. Remove the junction once nothing refers to it.
The reference tool's `adopt` does this for a download, after checking every file's size and sha256 against its provider.
