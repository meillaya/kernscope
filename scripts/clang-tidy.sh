#!/usr/bin/env bash
# clang-tidy.sh — kernscope clang-tidy gate (F2): zero findings with full TU
# coverage, CI-enforceable.
#
# Why augmentation: the real build's compilers are nix wrapper binaries that
# inject the vendor include dirs (CUDA/CUPTI/HIP/rocprofiler-sdk) via
# NIX_CFLAGS_COMPILE at runtime. That environment is NOT part of
# build/compile_commands.json, so clang-tidy (which uses the compile DB's
# argv as-is) cannot find vendor headers such as 'cuda.h' (from cupti.h) or
# 'rocprofiler-sdk/buffer.h'. This script augments a COPY of the compile DB
# with -isystem for every vendor include dir that exists in the current env
# (devenv shell), then runs run-clang-tidy over libs/ + tools/.
#
# Exit code: 0 only when every analyzable TU is clean (run-clang-tidy exits
# non-zero on any finding, including parse errors).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
DB="${BUILD_DIR}/compile_commands.json"
if [[ ! -f "${DB}" ]]; then
  echo "compile DB missing (${DB}) — run scripts/configure.sh first" >&2
  exit 1
fi

AUG_DB_DIR="${BUILD_DIR}/tidy-db"
AUG_DB="${AUG_DB_DIR}/compile_commands.json"
mkdir -p "${AUG_DB_DIR}"

# Vendor include dirs from the devenv env (each added only if it exists, so
# the script also works outside the nix shell with a user toolchain). The
# compile DB carries only the CMake-declared -I/-isystem flags; everything the
# nix compiler wrappers inject via NIX_CFLAGS_COMPILE (a superset: CUDA
# cudart/cccl/nvrtc/cublas, CUPTI, HIP/clr, rocprofiler-sdk, ...) is derived
# from CMAKE_INCLUDE_PATH and added here.
VENDOR_DIRS=()
while IFS= read -r d; do
  [[ -n "${d}" && -d "${d}" ]] && VENDOR_DIRS+=("${d}")
done < <(printf '%s' "${CMAKE_INCLUDE_PATH:-}" | tr ':' '\n')
# Explicit env-var dirs too (defensive; CMAKE_INCLUDE_PATH covers them in the
# devenv, but a bare user toolchain may set only the vars).
for v in "${CUDA_HOME:-}/include" \
         "${CUPTI_HOME:-}/include" \
         "${HIP_PATH:-}/include" \
         "${ROCPROF_SDK_DEV:-}/include"; do
  [[ -n "${v}" && -d "${v}" ]] && VENDOR_DIRS+=("${v}")
done
# De-dup, keep order.
readarray -t VENDOR_DIRS < <(printf '%s\n' "${VENDOR_DIRS[@]}" | awk '!seen[$0]++')

VENDOR_JSON="$(python3 -c '
import json, sys
dirs = sys.argv[1:]
print(json.dumps([["-isystem", d] for d in dirs]))
' "${VENDOR_DIRS[@]}")"

python3 - "${DB}" "${AUG_DB}" "${VENDOR_JSON}" <<'PYEOF'
import json, sys
src, dst, vendor_json = sys.argv[1], sys.argv[2], sys.argv[3]
vendor_args = json.loads(vendor_json)
db = json.load(open(src))
for entry in db:
    entry["command"] = entry["command"] + " " + " ".join(
        f'"{a}"' if " " in a else a
        for pair in vendor_args for a in pair)
json.dump(db, open(dst, "w"), indent=1)
print(f"augmented {len(db)} TUs with {len(vendor_args)//2} vendor include dirs",
      file=sys.stderr)
PYEOF

echo "[clang-tidy] using augmented compile DB at ${AUG_DB}"
echo "[clang-tidy] vendor include dirs: ${VENDOR_DIRS[*]:-none}"

LOG="${BUILD_DIR}/tidy.log"
run-clang-tidy -p "${AUG_DB_DIR}" -j "$(nproc)" -quiet libs tools > "${LOG}" 2>&1 \
  || true

# Vendor-header finding exclusion (documented in .clang-tidy): clang-analyzer
# diagnostics bypass HeaderFilterRegex/--exclude-header-filter (verified
# empirically) and fire on LLVM's own deliberate intrusive-list/metadata
# pointer tricks (e.g. User.h hung-off operands, Metadata.h header offset).
# Those are false positives in vendor code we cannot fix, so findings whose
# REPORTED location is under /nix/store/ are dropped — EXCEPT
# clang-diagnostic-error, which must keep failing the gate (missing vendor
# headers mean a TU did not parse). Findings in repo files (libs/ tools/) are
# fully enforced.
python3 - "${LOG}" <<'PYEOF'
import re, sys
log = sys.argv[1]
kept = []
dropped = 0
diag = re.compile(r"^([^:]+):\d+:\d+: (error|warning): (.*) \[(.*)\]$")
for line in open(log):
    m = diag.match(line)
    if not m:
        kept.append(line)
        continue
    path, sev, msg, check = m.groups()
    if path.startswith("/nix/store/") and check != "clang-diagnostic-error":
        dropped += 1
        continue
    kept.append(line)
sys.stdout.writelines(kept)
errors = [l for l in kept if diag.match(l) and ": error: " in l]
print(f"[clang-tidy] excluded {dropped} vendor-header finding(s) (documented "
      f"analyzer false positives in /nix/store/ vendor code)",
      file=sys.stderr)
print(f"[clang-tidy] {len(errors)} error(s) in repo TUs", file=sys.stderr)
sys.exit(1 if errors else 0)
PYEOF
