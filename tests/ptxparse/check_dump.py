#!/usr/bin/env python3
"""ctest gate for kernscope-ptxdump JSON output.

Usage: check_dump.py <ptxdump-binary> <file.ptx>

Runs `kernscope-ptxdump <file.ptx>` (exit 0 required) and asserts the emitted
JSON carries the todo-9 acceptance fields:

  * a non-empty "entry_names" list (entry-name present)
  * kernels[].name non-empty with entry == true
  * kernels[].instruction_count > 0
  * kernels[].bar_sync_count >= 1
  * kernels[].shared_access_count >= 1   (real fixture must have .shared ops)
  * kernels[].global_access_count >= 1
  * kernels[].tensor_ops is a list

Exit 0 on success, non-zero with a message on any violation.
"""

import json
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <ptxdump-binary> <file.ptx>", file=sys.stderr)
        return 2
    tool, ptx = sys.argv[1], sys.argv[2]

    proc = subprocess.run([tool, ptx], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"kernscope-ptxdump exited {proc.returncode} on {ptx}", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        return 1

    try:
        doc = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"output is not valid JSON: {exc}", file=sys.stderr)
        return 1

    failures = []

    entry_names = doc.get("entry_names")
    if not isinstance(entry_names, list) or not entry_names:
        failures.append('missing non-empty "entry_names"')
    if doc.get("ptx_version") != "8.5":
        failures.append(f'ptx_version != "8.5" (got {doc.get("ptx_version")!r})')

    kernels = doc.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        failures.append('missing non-empty "kernels"')
        print(f"check_dump FAIL: {'; '.join(failures)}", file=sys.stderr)
        return 1

    k = kernels[0]
    checks = [
        ("kernel name non-empty", bool(k.get("name"))),
        ("kernel entry flag", k.get("entry") is True),
        ("instruction_count > 0", k.get("instruction_count", 0) > 0),
        ("bar_sync_count >= 1", k.get("bar_sync_count", 0) >= 1),
        ("shared_access_count >= 1", k.get("shared_access_count", 0) >= 1),
        ("global_access_count >= 1", k.get("global_access_count", 0) >= 1),
        ("cfg_edge_count >= 0", k.get("cfg_edge_count", -1) >= 0),
        ("class_counts is a dict", isinstance(k.get("class_counts"), dict)),
        ("tensor_ops is a list", isinstance(k.get("tensor_ops"), list)),
    ]
    for label, ok in checks:
        if not ok:
            failures.append(label)

    if failures:
        print(f"check_dump FAIL: {'; '.join(failures)}", file=sys.stderr)
        return 1

    print(f"check_dump OK: {k['name']} instruction_count={k['instruction_count']} "
          f"bar_sync={k['bar_sync_count']} shared={k['shared_access_count']} "
          f"global={k['global_access_count']} cfg_edges={k['cfg_edge_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
