#!/usr/bin/env python3
"""Minimal JSON Schema draft-2020-12 subset validator (todo 7, kernscope).

Validates a JSON instance file against a schema file. Implements exactly the
keyword subset the committed kernscope schemas use (docs/schema/*.json):

    $ref, type, required, properties, additionalProperties, items,
    enum, const

plus the annotation-only keywords $schema/$id/title/description/$defs
($defs/definitions are only used as $ref targets). Unknown keywords are
ignored (draft-compliant behavior for a subset validator).

Usage:
    python3 validate_json.py <instance.json> <schema.json>
Exit 0 = valid, 1 = invalid (first violation printed to stderr),
2 = usage/IO error.
"""

import json
import sys


def _err(msg):
    sys.stderr.write("validate_json: %s\n" % msg)


def resolve_ref(ref, root_schema):
    """Resolve a JSON-pointer $ref against the root schema. Only #/$defs/...
    and #/definitions/... are needed by the kernscope schemas."""
    if not ref.startswith("#/"):
        raise ValueError("unsupported $ref %r (only local pointers)" % ref)
    parts = [p.replace("~1", "/").replace("~0", "~") for p in ref[2:].split("/")]
    node = root_schema
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            raise ValueError("dangling $ref %r" % ref)
        node = node[p]
    return node


def check(instance, schema, root_schema, path, errors):
    if not isinstance(schema, dict):
        return
    if "$ref" in schema:
        try:
            target = resolve_ref(schema["$ref"], root_schema)
        except ValueError as e:
            errors.append("%s: %s" % (path, e))
            return
        check(instance, target, root_schema, path, errors)
        return

    # --- type ---
    if "type" in schema:
        want = schema["type"]
        types = want if isinstance(want, list) else [want]
        ok = False
        for t in types:
            if t == "object" and isinstance(instance, dict):
                ok = True
            elif t == "array" and isinstance(instance, list):
                ok = True
            elif t == "string" and isinstance(instance, str):
                ok = True
            elif t == "number" and isinstance(instance, (int, float)) \
                    and not isinstance(instance, bool):
                ok = True
            elif t == "integer" and isinstance(instance, int) \
                    and not isinstance(instance, bool):
                ok = True
            elif t == "boolean" and isinstance(instance, bool):
                ok = True
            elif t == "null" and instance is None:
                ok = True
        if not ok:
            errors.append("%s: expected type %s, got %s"
                          % (path, want, type(instance).__name__))
            return

    # --- enum / const ---
    if "enum" in schema and instance not in schema["enum"]:
        errors.append("%s: value not in enum %s" % (path, schema["enum"]))
        return
    if "const" in schema and instance != schema["const"]:
        errors.append("%s: expected const %r, got %r"
                      % (path, schema["const"], instance))
        return

    # --- object keywords ---
    if isinstance(instance, dict):
        if "required" in schema:
            for key in schema["required"]:
                if key not in instance:
                    errors.append("%s: missing required property %r" % (path, key))
        if "properties" in schema:
            for key, subschema in schema["properties"].items():
                if key in instance:
                    check(instance[key], subschema, root_schema,
                          "%s.%s" % (path, key), errors)
        if "additionalProperties" in schema:
            addl = schema["additionalProperties"]
            known = set(schema.get("properties", {}))
            for key in instance:
                if key not in known:
                    if addl is True:
                        continue
                    if addl is False:
                        errors.append("%s: additional property %r not allowed"
                                      % (path, key))
                    else:
                        check(instance[key], addl, root_schema,
                              "%s.%s" % (path, key), errors)

    # --- array keywords ---
    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            check(item, schema["items"], root_schema,
                  "%s[%d]" % (path, i), errors)


def main():
    if len(sys.argv) != 3:
        _err("usage: validate_json.py <instance.json> <schema.json>")
        return 2
    try:
        with open(sys.argv[1], encoding="utf-8") as f:
            instance = json.load(f)
        with open(sys.argv[2], encoding="utf-8") as f:
            schema = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        _err("load failed: %s" % e)
        return 2

    errors = []
    check(instance, schema, schema, "$", errors)
    if errors:
        for e in errors[:20]:
            _err(e)
        if len(errors) > 20:
            _err("... and %d more" % (len(errors) - 20))
        return 1
    print("validate_json: %s valid against %s"
          % (sys.argv[1], sys.argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
