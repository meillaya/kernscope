// kernscope-ptxdump: JSON dump of a parsed PTX module.
//
// Usage: kernscope-ptxdump <file.ptx>
//
// Emits one JSON document to stdout: directives (.version/.target), one
// object per kernel (.entry/.func) with the name, entry flag, instruction
// count, CFG edge count, bar.sync count, per-state-space access counts
// (.shared/.global/.local/.const/.param), ld/st width histogram, the
// instruction-mix class counts, and the tensor-op list (class tag +
// parse-only annotation for hand-written tcgen05 families).
//
// Exit status: 0 on success; 1 when the input is malformed (a line:col
// diagnostic is printed to stderr - never a silent partial dump); 2 on usage
// errors.
//
// The JSON value model is kernscope::db::json (self-contained, canonical
// serialization, gtest-covered in libs/db) - no external JSON dependency.

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "kernscope/db.h"
#include "kernscope/ptx.h"

namespace {

void print_usage(FILE *out) {
  std::fprintf(out, "usage: kernscope-ptxdump <file.ptx>\n");
  std::fprintf(out, "  emits a JSON summary of the parsed PTX module\n");
}

kernscope::db::json::Value tensor_op_json(const kernscope::PtxTensorOp &op) {
  return kernscope::db::json::object(
      {{"mnemonic", kernscope::db::json::Value(op.mnemonic)},
       {"class", kernscope::db::json::Value(op.class_tag)},
       {"parse_only", kernscope::db::json::Value(op.parse_only)}});
}

kernscope::db::json::Value kernel_json(const kernscope::PtxKernel &k) {
  std::vector<kernscope::db::json::Value::Member> members;
  members.emplace_back("name", kernscope::db::json::Value(k.name));
  members.emplace_back("entry", kernscope::db::json::Value(k.entry_point));
  members.emplace_back("instruction_count",
                       kernscope::db::json::Value(
                           static_cast<int>(k.instruction_count)));
  members.emplace_back("cfg_edge_count",
                       kernscope::db::json::Value(static_cast<int>(k.cfg_edge_count)));
  members.emplace_back("cfg_block_count",
                       kernscope::db::json::Value(static_cast<int>(k.cfg.size())));
  members.emplace_back("bar_sync_count",
                       kernscope::db::json::Value(static_cast<int>(k.bar_sync_count)));
  members.emplace_back("shared_access_count",
                       kernscope::db::json::Value(static_cast<int>(k.shared_access_count)));
  members.emplace_back("global_access_count",
                       kernscope::db::json::Value(static_cast<int>(k.global_access_count)));
  members.emplace_back("local_access_count",
                       kernscope::db::json::Value(static_cast<int>(k.local_access_count)));
  members.emplace_back("const_access_count",
                       kernscope::db::json::Value(static_cast<int>(k.const_access_count)));
  members.emplace_back("param_access_count",
                       kernscope::db::json::Value(static_cast<int>(k.param_access_count)));

  std::vector<kernscope::db::json::Value::Member> widths;
  widths.reserve(k.mem_width_counts.size());
  for (const auto &[bits, count] : k.mem_width_counts) {
    widths.emplace_back(std::to_string(bits),
                        kernscope::db::json::Value(static_cast<int>(count)));
  }
  members.emplace_back("mem_width_counts",
                       kernscope::db::json::Value(std::move(widths)));

  std::vector<kernscope::db::json::Value::Member> classes;
  classes.reserve(k.class_counts.size());
  for (const auto &[cls, count] : k.class_counts) {
    classes.emplace_back(
        kernscope::instruction_class_name(cls),
        kernscope::db::json::Value(static_cast<int>(count)));
  }
  members.emplace_back("class_counts",
                       kernscope::db::json::Value(std::move(classes)));

  std::vector<kernscope::db::json::Value> tensors;
  tensors.reserve(k.tensor_ops.size());
  for (const auto &op : k.tensor_ops) tensors.push_back(tensor_op_json(op));
  members.emplace_back("tensor_ops",
                       kernscope::db::json::Value(std::move(tensors)));

  return kernscope::db::json::Value(std::move(members));
}

bool read_file(const std::string &path, std::string &out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  out = ss.str();
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    print_usage(stderr);
    return 2;
  }
  const std::string path = argv[1];

  std::string text;
  if (!read_file(path, text)) {
    std::fprintf(stderr, "kernscope-ptxdump: cannot open '%s'\n", path.c_str());
    return 2;
  }

  kernscope::PtxModule mod;
  try {
    mod = kernscope::parse_ptx_module(text);
  } catch (const kernscope::PtxParseError &e) {
    std::fprintf(stderr, "kernscope-ptxdump: %s\n", e.what());
    return 1;
  }

  std::vector<kernscope::db::json::Value> kernels;
  std::vector<kernscope::db::json::Value> entry_names;
  for (const auto &k : mod.kernels()) {
    kernels.push_back(kernel_json(k));
    if (k.entry_point) entry_names.emplace_back(k.name);
  }

  const auto doc = kernscope::db::json::object(
      {{"file", kernscope::db::json::Value(path)},
       {"ptx_version", kernscope::db::json::Value(mod.ptx_version)},
       {"target", kernscope::db::json::Value(mod.target)},
       {"address_size", kernscope::db::json::Value(mod.address_size)},
       {"entry_names", kernscope::db::json::Value(std::move(entry_names))},
       {"kernels", kernscope::db::json::Value(std::move(kernels))}});

  std::puts(doc.dump().c_str());
  return 0;
}
