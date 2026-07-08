# Tokenizer Caching

Loading a tokenizer directly from `tokenizer.json` requires re-parsing the
entire vocabulary, merge list, and normalizer rules on every application
start. For WordPiece and BPE tokenizers, CausalLM avoids this cost: after
the first load, a compact binary snapshot is written next to the tokenizer
file, and subsequent loads read the snapshot instead of re-parsing the
source JSON.

## Setup

No configuration is required. If `nntr_config.json` specifies a
`tokenizer_file` pointing to a WordPiece tokenizer (`vocab.txt`, or a
WordPiece-style `tokenizer.json`) or a BPE tokenizer (`tokenizer.json`),
caching is applied automatically:

- **First load**: the tokenizer is built normally, and a cache file is
  written next to `tokenizer_file`.
- **Subsequent loads**: the cache file is read instead, making tokenizer
  initialization effectively instantaneous.

Tokenizer behavior is unaffected — encoding and decoding results are
identical whether or not the cache is used. Tokenizer formats other than
WordPiece and BPE (e.g. SentencePiece) are unaffected and continue to load
through the standard path.

## Verifying that caching is active

A new file should appear next to `tokenizer_file` after the first load:

```
tokenizer.json
tokenizer.json.qaibpe   <- BPE cache, created after the first load
```

or, for a WordPiece `vocab.txt`:

```
vocab.txt
vocab.txt.qaiwp         <- WordPiece cache
```

If this file exists and its modification time is at least as recent as
the tokenizer file it was generated from, the cache is in use.

## Supported tokenizer types

Caching applies only to the two formats implemented in this directory:

| Format | `tokenizer_type` values | Source file(s) recognized | Backing implementation |
| --- | --- | --- | --- |
| WordPiece | `"wordpiece"`, `"bert"`, `"tinybert"` | A `vocab.txt`-style file (one token per line), or a `tokenizer.json` whose `model.type` is `"WordPiece"` | `wordpiece_tokenizer.cpp` |
| BPE | `"bpe"`, `"bytelevelbpe"` | A `tokenizer.json` whose `model.type` is `"BPE"` | `bpe_tokenizer.cpp` |

`tokenizer_type` values are matched case-insensitively. When `tokenizer_type`
is not set, the format is inferred from `tokenizer_file`: a `.txt` file or a
filename containing `vocab` is treated as WordPiece; a `tokenizer.json` is
classified by reading its own `model.type` field directly (the same field
the `tokenizers` library itself uses to decide how to parse the file), so
detection matches the file's actual format rather than a filename or
content heuristic. All other tokenizer formats (SentencePiece, RWKV World,
generic byte-level BPE via `FromBlobByteLevelBPE`, or a `tokenizer.json`
whose `model.type` is neither `WordPiece` nor `BPE`) are handled by
`huggingface_tokenizer.cpp` and are not cached.

### Tokenizer format by model

The tokenizer format is always determined by inspecting the `tokenizer.json`
(or `vocab.txt`) that ships with a given checkpoint, not by which model
architecture is loading it. The table below reflects the format used by the
officially published checkpoints for each model family currently supported
by CausalLM, as a practical reference; substituting a different checkpoint
for the same architecture is cached or not based on that checkpoint's own
tokenizer file, not the entries below.

| Model family | Tokenizer format | Cached |
| --- | --- | --- |
| Llama (`causal_lm`) | BPE | Yes |
| Qwen2 | BPE | Yes |
| Qwen3 / Qwen3-MoE / Qwen3-Slim-MoE / Qwen3-Cached-Slim-MoE | BPE | Yes |
| GPT-OSS / GPT-OSS-Cached-Slim | BPE | Yes |
| Gemma3 / Gemma4 | BPE | Yes |
| KaLM-Embedding (Qwen2 backbone) | BPE | Yes |
| TinyBERT / multilingual TinyBERT (`bert`) | WordPiece | Yes |
| DeBERTa-v2 | SentencePiece (Unigram) | No — handled by `huggingface_tokenizer.cpp` |
| TimmViT | N/A (vision encoder, no tokenizer) | N/A |

## Configuration

The following keys are read from `nntr_config.json`, alongside
`tokenizer_file`:

| Key | Required | Description |
| --- | --- | --- |
| `tokenizer_file` | Yes | Path to the `tokenizer.json` or `vocab.txt` file. |
| `tokenizer_type` | No | Forces `"wordpiece"` (or `"bert"` / `"tinybert"`) or `"bpe"` (or `"bytelevelbpe"`). If omitted, the type is detected automatically. |
| `tokenizer_cache` | No | Enables or disables caching entirely. Defaults to `true`. Set to `false` to always load from `tokenizer_file` and skip reading or writing a cache file. |
| `tokenizer_cache_file` | No | Overrides the cache file path. Defaults to `<tokenizer_file>.qaiwp` (WordPiece) or `<tokenizer_file>.qaibpe` (BPE). Ignored when `tokenizer_cache` is `false`. |
| `tokenizer_do_lower_case`, `tokenizer_unk_token`, `tokenizer_continuing_subword_prefix`, `tokenizer_max_input_chars_per_word`, `tokenizer_cls_token`, `tokenizer_sep_token` | No | WordPiece configuration overrides. Required only to override values already present in the tokenizer file. |

Example, forcing WordPiece explicitly:

```json
{
  "tokenizer_file": "vocab.txt",
  "tokenizer_type": "wordpiece",
  "tokenizer_do_lower_case": true
}
```

`tokenizer_type` is required only when automatic detection needs to be
overridden; in most configurations it may be omitted.

Example, disabling caching:

```json
{
  "tokenizer_file": "tokenizer.json",
  "tokenizer_cache": false
}
```

This is useful when `tokenizer_file` lives on a read-only filesystem, or
when a test/CI environment should not leave cache files behind as a side
effect of running.

## Resetting the cache

If the tokenizer file is edited or replaced in place and the change does
not appear to take effect, delete the cache file to force a rebuild on
the next load:

```bash
rm tokenizer.json.qaibpe   # or vocab.txt.qaiwp
```

This step is not required under normal operation: the cache is rebuilt
automatically whenever it is older than the tokenizer file, or whenever it
fails to load (for example, due to corruption or a format mismatch).
Manual deletion simply forces an immediate rebuild. To stop cache files
from being written at all, set `tokenizer_cache` to `false` instead (see
Configuration above).

## Troubleshooting

- **No cache file is created after running.** The tokenizer is likely not
  recognized as WordPiece or BPE (for example, it is a SentencePiece
  model, or the `tokenizer.json` does not match either format). This is
  expected behavior: such tokenizers always load through the standard
  path and do not use a cache. If automatic detection appears incorrect,
  set `tokenizer_type` explicitly.
- **The tokenizer file was changed, but behavior does not reflect the
  update.** Delete the cache file as described above and reload.
- **Can caching change encoding results?** No. For BPE, before the
  cached or native path is trusted, its output is verified against the
  standard tokenizer across a battery of conformance prompts, and the
  fast path is used only if the results match exactly. On any mismatch,
  the standard tokenizer is used instead, and the cache is not applied.
  WordPiece caching is a direct re-encoding of the vocabulary file, so no
  mismatch is possible.

## Implementation notes

<details>
<summary>File map, load order, and cache format</summary>

**Files in this directory:**

- `huggingface_tokenizer.cpp` — wraps the upstream `tokenizers-cpp` /
  HuggingFace `tokenizers` library. This is the fallback used for any
  tokenizer that is not WordPiece or BPE, or for which the faster path
  cannot be verified.
- `bpe_tokenizer.cpp` / `wordpiece_tokenizer.cpp` — native tokenizer
  implementations capable of (de)serializing to and from the binary
  cache blob.
- `tokenizer_cache_util.h` — shared binary encode/decode helpers (magic
  bytes, version, endian-safe integer read/write) used by both cache
  formats.
- `../models/tokenizer_loader.{h,cpp}` — defines `causallm::LoadTokenizer()`,
  the entry point called by `Transformer` in place of interacting with
  `tokenizers::Tokenizer` directly. This function selects the backend to
  use and manages cache read/write.

**Load order for a given `tokenizer_file`:**

1. If `tokenizer_cache` is not `false`, and a cache file exists and is not
   older than `tokenizer_file` (based on an `mtime` comparison), it is
   deserialized directly, without any JSON parsing.
2. Otherwise, the tokenizer is built from source. The format is chosen by
   reading `tokenizer.json`'s own `model.type` field (or, for a non-JSON
   file, by filename), not by `tokenizer_type` or any content heuristic.
   For BPE, the native tokenizer's output is cross-checked against the
   standard HF tokenizer over a battery of conformance prompts before
   being trusted; on any mismatch, the native result is discarded and the
   HF tokenizer is used instead. WordPiece requires no such check, as the
   vocabulary format is unambiguous.
3. If a native tokenizer is accepted and `tokenizer_cache` is not `false`,
   it is serialized and the cache file is written for subsequent loads.
4. If the tokenizer type cannot be identified as WordPiece or BPE,
   `LoadTokenizer` falls back directly to
   `tokenizers::Tokenizer::FromBlobJSON()`, the same path used prior to
   the introduction of caching.

**Cache blob header** (`tokenizer_cache_util.h`):

```
[8 bytes magic "QAITOKCH"] [u32 version] [u32 kind] [format-specific payload]
```

`kind` distinguishes `CacheKind::WordPiece` (1) from `CacheKind::BPE` (2);
a mismatch is treated as a corrupt cache. The payload layout is private to
each tokenizer implementation and is not guaranteed to remain stable
across `kCacheVersion` increments. Any parse failure (invalid magic,
version mismatch, truncated data) is caught, logged to stderr, and
treated as a cache miss.

</details>
