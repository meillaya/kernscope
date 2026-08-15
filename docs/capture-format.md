# kernscope Capture Bundle Format — v1

Status: todo 46 deliverable (bundle format + VA-faithful device-state snapshot +
size governance). Author: capture track. Consumed by todo 47 (reproducer
generator) and todo 48 (replay engine); the v0 bundles produced by todos
43/44/45 remain loadable (see §7).

The capture mechanisms (LD_PRELOAD shim `preload`, CUPTI `cupti`, HIP
rocprofiler-sdk `hip-sdk`) record into the shared `libs/capture-core` session
model and serialize through its bundle writer. **v1 is the formal format**:
a directory-based bundle (`*.kcap/`) carrying a JSON manifest, a formalized
event log with an arg marshalling table, and VA-faithful buffer blobs with
sha256 provenance and embedded-pointer patch tables. The reader
(`BundleReader`, libs/capture-core) loads v1 bundles AND the v0 line-oriented
text bundles todos 43/44/45 still emit (§7 — the compatibility decision).

## 1. Bundle layout

A v1 bundle is a directory ending in `.kcap` (gitignored repo-wide, see
§5.2):

```
<name>.kcap/
  manifest.json    # envelope: format id, schema version, capture metadata, size governance
  events.json      # the formalized event log, incl. the per-launch arg marshalling table
  blobs.json       # blob index: provenance hash + original VA + patch table per blob
  blobs/
    b0001.bin      # one file per blob (kernel binary, input/output buffers, ...)
    b0002.bin
```

All JSON in the bundle is produced with the repo's canonical serializer
(`kernscope::db::json::Value::dump`, todo 8 conventions): object members in
lexicographic key order, numbers in shortest round-trip form → byte-identical
output for identical logical bundles.

### 1.1 64-bit integer convention (u64 → hex string)

All 64-bit integer fields — pointer values, **original virtual addresses**,
byte sizes, stream/graph handles, blob offsets — are serialized as lowercase
hex strings with a `0x` prefix (e.g. `"0x400"` = 1024). JSON numbers are IEEE
doubles and cannot represent every u64 exactly above 2^53; hex strings are
exact. Small counters (grid/block dims, `calls`, `seq`, `cuda_kind`,
`param_index`) are JSON numbers. Every schema field says which it is.

## 2. manifest.json

```json
{
  "format_id": "kernscope-capture",
  "schema_version": "1.0",
  "tool_version": "kernscope-capture/1.0",
  "vendor": "mock",
  "arch": "sm_80",
  "capture_mechanism": "preload",
  "timestamp": "2026-08-14T04:00:00Z",
  "snapshot_policy": "va-faithful",
  "size_governance": {
    "max_snapshot_bytes": 1073741824,
    "total_blob_bytes": 1088,
    "aborted": false,
    "abort_reason": ""
  },
  "first_kernel": "vec_add"
}
```

| field | type | meaning |
|---|---|---|
| `format_id` | string | always `"kernscope-capture"`; the reader refuses anything else |
| `schema_version` | string | `"1.0"` for v1 (see §8 versioning) |
| `tool_version` | string | producer tool version |
| `vendor` | string | `nvidia` \| `amd` \| `mock` (the equivalence test's M12 rule: never cross-vendor blob byte equality) |
| `arch` | string | `sm_80`, `gfx1101`, ... |
| `capture_mechanism` | string | `cupti` \| `preload` \| `hip-sdk` (the plan's enumerated set) |
| `timestamp` | string | ISO-8601 UTC, e.g. `"2026-08-14T04:00:00Z"` |
| `snapshot_policy` | string | always `"va-faithful"` in v1 (Kerncap policy, research brief §5.3: snapshot tracked allocations **at their original VAs**; replay restores at original VAs where the platform allows, else remaps + patches — §4.4) |
| `size_governance` | object | §5. `aborted=true` + non-empty `abort_reason` marks a **partial** bundle (size-cap abort; §5.1) |
| `first_kernel` | string | first launch record's kernel name, if any |

## 3. events.json — the formalized event log

One JSON array of event objects, in record order. The event types mirror the
v0 line tokens 1:1 (record semantics unchanged — §7.2); v1 only adds the
`seq` field (monotonic event order, already in the session model) and the
`param_info` records the v0 writer dropped. u64 fields are hex strings
(§1.1).

```
api           {"type":"api","name":"cudaMalloc","calls":1}
alloc         {"type":"alloc","ptr":"0x1000","size":"0x400","is_alloc":true,"seq":1}
free          {"type":"free","ptr":"0x1000","size":"0x400","is_alloc":false,"seq":2}
unknown-free  {"type":"unknown-free","ptr":"0xbeef","forwarded":true,"seq":3}
launch        {"type":"launch","kernel_name":"vec_add","via":"runtime",
               "grid":[2,1,1],"block":[128,1,1],"shared_mem_bytes":0,
               "stream":"0x0","graph_handle":"0x0","seq":4,
               "args":[{arg slot...}, ...]}
copy          {"type":"copy","kind":"memcpy","dst":"0x2000","src":"0x7f..",
               "bytes":"0x200","cuda_kind":1,"dst_size":"0x400","src_size":"0x0",
               "stream":"0x0","seq":5}
memset        {"type":"memset","ptr":"0x3000","value":0,"bytes":"0x100",
               "ptr_size":"0x100","stream":"0x0","seq":6}
stream        {"type":"stream","sync":false,"handle":"0x1234","id":1,"seq":7}
fatbin        {"type":"fatbin","mod":1,"sha256":"<64 hex>","kernels":["vec_add"],"seq":8}
cubin         {"type":"cubin","mod":2,"sha256":"<64 hex>","path":"","kernels":[],"seq":9}
param_info    {"type":"param_info","cufunction":"0x9000","param_index":0,
               "offset":0,"size":8,"seq":10}
```

`via` values: `runtime` | `runtime-ex` | `graph` | `driver` | `driver-ex`
(the `to_string(LaunchVia)` vocabulary). `cuda_kind` is the cudaMemcpyKind
value (1 H2D, 2 D2H, 3 D2D — docs/capture-mapping.md M10). `fatbin`/`cubin`
keep the v0 line tokens as the `type` so the record semantics todos 43/44/45
produce are unchanged.

### 3.1 The arg marshalling table (per-launch `args` array)

The plan's "arg marshalling table (scalar/by-value bytes + pointer slots)"
is the `args` array of each launch event. One object per kernel parameter,
in parameter order:

```
{"kind":"pointer","size":8,"bytes":"0010000000000000","va":"0x1000"}
{"kind":"scalar", "size":4,"bytes":"00020000"}
```

| field | type | meaning |
|---|---|---|
| `kind` | string | `scalar` (by-value bytes) or `pointer` (device-pointer slot) |
| `size` | number | parameter size in bytes (cuFuncGetParamInfo / AMDGPU-metadata layout; 0 = size unknown at the boundary) |
| `bytes` | string | the parameter's host-side bytes, lowercase hex (byte-exact copy contract, capture-mapping §2/M7) |
| `va` | string (hex) | pointer slots only: the **original device VA** (VA-faithful; capture-mapping §4.3's normalization input) |

**Pointer-slot classification rule** (documented heuristic, no type info
needed): an 8-byte slot whose little-endian value equals the VA of an
allocation in the session's alloc records is a `pointer` slot; everything
else is `scalar`. This is capture-mapping §4.3 applied at write time, and the
same rule the blob patch scanner uses (§4.3). Replay resolves a pointer slot
to the allocation ordinal (capture-mapping §4.3) or to the remapped VA
(§4.4).

## 4. blobs.json + blobs/ — VA-faithful snapshot

Every captured byte payload is a **blob**: kernel binaries (cubin/hsaco) and
input/output buffer snapshots. Blob bytes live in `blobs/<id>.bin`; every
blob is indexed in `blobs.json` with **mandatory provenance**: the sha256
digest of exactly the blob's bytes (FIPS 180-4, `sha256_hex`).

```json
{
  "schema_version": "1.0",
  "blobs": [
    {"id":"b0001","file":"blobs/b0001.bin","role":"kernel-binary",
     "va":"0x0","size":"0x3","sha256":"<64 hex>","patches":[]},
    {"id":"b0002","file":"blobs/b0002.bin","role":"input",
     "va":"0x2000","size":"0x40","sha256":"<64 hex>",
     "patches":[{"offset":16,"target_va":"0x2000"}]}
  ]
}
```

| field | type | meaning |
|---|---|---|
| `id` | string | `b0001`... assigned in write order |
| `file` | string | path relative to the bundle dir |
| `role` | string | `kernel-binary` \| `input` \| `output` \| `arg` \| `state` |
| `va` | string (hex) | **original virtual address** of the buffer (VA-faithful; `0x0` for blobs with no device VA, e.g. cubins) |
| `size` | string (hex) | blob byte count |
| `sha256` | string | mandatory provenance hash of the blob bytes; the reader **verifies** it on load (mismatch → schema violation, §6) |
| `patches` | array | the embedded-pointer **patch table** (§4.3), empty when the scan found no candidates |

### 4.1 The snapshot policy (VA-faithful, per Kerncap)

Kerncap's address-space closure (research brief §5.3) snapshots **all
tracked allocations at their original VAs**, which makes embedded pointer
graphs (`T**`, vLLM MoE indirection) replay correctly with zero type info.
v1 adopts this as the policy: the writer records each snapped buffer with
its `va` as the platform assigned it. Replay (todo 48) restores snapshots at
the original VAs where the platform allows; where it does not, it remaps the
snapshot to fresh memory and applies the patch table (§4.4).

Sizes come from the session's ptr→size allocation map (GPGPU-Sim
`g_mallocPtr_Size` pattern, brief §5.1) — captured buffers are attributed
their EXACT byte counts, never guessed.

### 4.2 Mandatory provenance

A blob MUST NOT be stored without its sha256 (§6). The writer computes it
from the exact bytes written; the reader recomputes and compares on every
`read_blob` (and rejects on mismatch). The bundle thus self-authenticates
against silent corruption — the same contract todo 43 established for
cubin/fatbin hashes (capture-mapping M13: module sha256 == blob sha256
bijection).

### 4.3 The embedded-pointer patch table (T**)

Deep pointers inside a snapped buffer (a `T**` indirection table, an array of
device pointers, ...) must be rewritten when replay remaps the snapshot. The
writer runs a deterministic **scan heuristic** over each blob: every byte
offset is checked for an 8-byte little-endian value that exactly equals the
VA of a known allocation (from the session's alloc records); each hit becomes
a patch candidate `{offset, target_va}`. Candidates are recorded in
`patches` — they are exactly the deep-pointer slots, found with zero type
info (Kerncap's argument for whole-address-space closure).

`apply_remap(blob, patches, remap)` (libs/capture-core, pure + unit-tested
with synthetic VA tables) rewrites each candidate whose `target_va` appears
in the remap table to its new VA in place; candidates whose target was not
remapped are left untouched. The patch table is a *candidate* table by
design: only remap-table hits are written.

### 4.4 Replay contract (todo 48, specified here)

1. Restore each snapshot at its **original VA** where the platform allows
   (Kerncap's path); then no patching is needed.
2. Else (remap): allocate fresh memory, load the blob, and for every patch
   entry with `remap[target_va]` set, write the new VA at `offset` (this
   must happen BEFORE the launch reads the buffer — the test asserts it on
   the bytes).
3. Pointer **arg slots** (§3.1) are remapped through the same remap table.
4. Mneme's documented failure case — a `__device__` global holding another
   global's address (brief §5) — is documented as out of scope: the device
   global's VA is not a captured allocation, so the scan does not see it.

## 5. Size governance

### 5.1 --max-snapshot-bytes (default 1 GiB)

The writer enforces a snapshot byte budget: `BundleWriteOptions
.max_snapshot_bytes` **defaults to 1 GiB**
(`kDefaultMaxSnapshotBytes`, libs/capture-core; the plan pins 1 GiB default,
configurable). The budget is the sum of all blob bytes (`size_governance
.total_blob_bytes`). When the next blob would push the total over the cap,
the write **aborts**: no further blobs are written, `manifest.json` is
updated with `size_governance.aborted=true` and an `abort_reason` naming the
cap and the exceeded total, and the write returns `WriteStatus::
kAbortedSizeCap` with the same clear error. The partial bundle stays
loadable (reader reports `aborted()`); it is a deliberate partial
capture, not a corrupt one.

The future `kernscope-capture` CLI (todo 48) exposes this as
`--max-snapshot-bytes <N>`; the library option is what tests drive today.
Budget check happens before each blob write, so a bundle never contains a
blob that alone exceeds the cap.

### 5.2 Bundles are gitignored

`*.kcap/` is gitignored repo-wide (`.gitignore`): capture artifacts are
large, machine-generated provenance, not source. Committed test **fixtures**
stay under `tests/fixtures/` and are subject to the ci.sh guard: total
fixture bytes ≤ 64 MiB (`du -sm tests/fixtures`, todo 4). The v0 sample
fixture added with this todo is ~1.5 KiB.

## 6. Provenance & the reader's error model

* Every blob has a sha256; the reader verifies it and reports a schema
  violation on mismatch (no silent corruption).
* Every JSON file is parsed with `kernscope::db::json::Value::parse`, whose
  errors are **byte-offset** descriptions (`"byte N: ..."`, todo 8
  conventions). A truncated/corrupt bundle therefore fails with a schema
  violation carrying the offending byte offset — no UB, no crash (QA
  negative, asserted in the test suite).
* The reader validates required fields with the same convention:
  `"missing required field \"x\""` / `"field \"x\" is not a string"`, etc.
* Load order: manifest → events → blob index; a failure in any of them
  rejects the bundle with the first error.

## 7. v0 → v1: the compatibility decision

**Decision: v1 is an upgrade layer, not a replacement.** `create_v0()` keeps
its exact deterministic text output (its smoke/equivalence assertions depend
on it), and the existing mechanisms (todo 43 shim, todo 44 cupti-sub, todo
45 capture-hip) keep emitting v0 bundles — this todo must not modify them.
`BundleReader` loads BOTH:

* **v1**: directory bundles (§1).
* **v0**: the line-oriented text shape todos 43/44/45 produce
  (`kernscope-capture-v0` … `end-capture-v0`; the todo-45 HIP tool emits it
  via `KERNSCOPE_HIP_BUNDLE`, the shim via `kernscope-capture.log`, the
  CUPTI path via the equivalent harness bundles). The v0 parser accepts
  every line grammar the v0 writer emits (api/alloc/free/unknown-free/
  launch/copy/memset/stream/fatbin/cubin) and reports truncated input with
  line+column offsets. v0 bundles carry no blobs in-band; when a v0 text is
  loaded from a directory that also holds `*.bin` sidecars (the todo-44
  equivalence bundles' `input-buffer-<n>.bin`), they are loaded as blobs
  with role `input` and hashed (capture-mapping M11).

New mechanisms emit v1 (`write_bundle_v1`); migrating the existing three is
a mechanical swap deferred to todo 48 (the CLI owns emission then), with
record semantics unchanged (the `type` tokens mirror the v0 line tokens).

### 7.1 Record semantics are frozen

The event types of §3 are exactly the v0 line tokens plus `seq`/`param_info`
enrichment. Todo 48's replay consumes the reader model (`BundleReader`
exposes the same record structs as `CaptureSession`), so a session
serialized as v0 today deserializes identically through the v1 reader —
the compatibility test asserts this on the committed v0 fixture.

## 8. Schema versioning

* `format_id` must be `"kernscope-capture"` — everything else is refused.
* `schema_version` in v1 is `"1.0"`; `blobs.json` carries its own
  `schema_version` (they can evolve independently).
* Readers are **forward-compatible by policy**: an unknown `schema_version`
  *greater* than the reader's is warning-skipped where the store allows
  (libs/db convention) — for bundles, the manifest version gates which
  fields may be read, and an unknown version is rejected with a clear error
  rather than mis-parsed. Version bumps require a reader test first
  (reader-compatibility tests are part of every format change).
* The companion JSON Schema (`docs/schema/capture-bundle.schema.json`,
  draft 2020-12) is the normative shape reference; the C++ reader is the
  enforcing implementation.

## 9. Relationship to the other capture docs

* `docs/capture-mapping.md` (todo 44): the VA-normalization + field
  correspondence rules; v1's pointer-slot classification and patch-table
  scan are §4.3 of that doc applied at write time.
* `docs/research/capture-replay-research-brief.md` §5 (buffer-size
  heuristics, VA-faithful Kerncap snapshots, deep-pointer marshalling) and
  §3 (address-space closure) are the design basis for §4.
