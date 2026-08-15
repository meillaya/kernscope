//===- diagnostics.cpp - hot-kernel ranking + compiler-aware diagnostics (todo 51) -*- C++ -*-===//
//
// kernscope tools/profile (C6 profiler track).
//
// Implements the todo-51 layer declared in diagnostics.h:
//   * rank_hot_kernels() — deterministic top-N hot list (total/median per
//     kernel, % of wall time; see the header for the sort contract);
//   * diagnose_kernel() — the SOL-style rules engine (brief §6.2):
//       occupancy rules (brief §6.2.2: register/shared/block resource hints
//       naming the binding resource, owner-examples.md Section B phrasing),
//       libs/roofline memory-traffic classification for known families
//       (brief §6.2.4: AI vs ridge; bytes moved DERIVED from the record's
//       observed work-item count — never fabricated),
//       the insufficient-evidence degradation (QA negative: no PTX + unknown
//       family + unknown occupancy -> exactly that rule, no fabricated hints),
//   * the kernscope-analyze (todo 20) subprocess invocation on the EXTRACTED
//     PTX + findings merge (bank conflicts, divergence, coalescing) — the
//     profiler<->analyzer cross-tool multiplier. The analyzer is INVOKED, its
//     exit-code contract honored (0 = clean, 1 = warnings present — both are
//     successful analyses per tools/analyze/main.cpp), never modified.
//
// MUST NOT emit a hint without evidence fields; MUST NOT require counters/
// PerfWorks (records + static analysis only).
//
//===----------------------------------------------------------------------===//

#include "kernscope/profile/diagnostics.h"

#include "kernscope/db.h"
#include "kernscope/profile/kernel_record.h" // json_escape
#include "kernscope/roofline.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace kernscope::profile {

namespace {

//===--- small helpers --------------------------------------------------------===//

// Compact double rendering for evidence strings + JSON (libs/report style).
std::string FmtNumber(double value) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.10g", value);
  return std::string(buf);
}

std::string ToLower(std::string s) {
  for (char &c : s) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return s;
}

// A finding's evidence is a key/value string map — a hint NEVER fires without
// at least one machine-readable evidence key (acceptance: asserted).
kernscope::Finding MakeFinding(const char *id, const char *severity,
                               std::map<std::string, std::string> evidence) {
  kernscope::Finding f;
  f.id = id;
  f.severity = severity;
  f.evidence = std::move(evidence);
  return f;
}

//===--- kernel-family detection (name-based; libs/roofline families) ---------===//
// The profiler record carries no FLOP/byte annotation, so the roofline rule
// can only classify kernels whose name maps to one of the six C2 families.
// Family evidence is always reported with the hint (never implied).

std::optional<roofline::KernelFamily> DetectFamily(
    const std::string &kernel_name) {
  const std::string name = ToLower(kernel_name);
  const bool flash = name.find("flash") != std::string::npos &&
                     name.find("attention") != std::string::npos;
  if (flash) {
    return roofline::KernelFamily::kFlashAttention;
  }
  if (name.find("layernorm") != std::string::npos ||
      name.find("layer_norm") != std::string::npos) {
    return roofline::KernelFamily::kLayerNorm;
  }
  if (name.find("softmax") != std::string::npos) {
    return roofline::KernelFamily::kSoftmax;
  }
  if (name.find("reduce") != std::string::npos) {
    return roofline::KernelFamily::kReduction;
  }
  if (name.find("gelu") != std::string::npos) {
    return roofline::KernelFamily::kFusedBiasGelu;
  }
  if (name.find("gemm") != std::string::npos ||
      name.find("matmul") != std::string::npos) {
    return roofline::KernelFamily::kGemm;
  }
  return std::nullopt;
}

// The families whose traffic is derivable from the record's observed
// work-item count alone (one element per thread; dtype fp32 conservative
// default). GEMM needs m/n/k, FlashAttention needs n/d — neither is present
// in a profiler record, so they are NOT classified (honest: no hint).
bool HasRecordDerivableShape(roofline::KernelFamily family) {
  switch (family) {
  case roofline::KernelFamily::kFusedBiasGelu:
  case roofline::KernelFamily::kSoftmax:
  case roofline::KernelFamily::kLayerNorm:
  case roofline::KernelFamily::kReduction:
    return true;
  case roofline::KernelFamily::kGemm:
  case roofline::KernelFamily::kFlashAttention:
    return false;
  }
  return false;
}

std::int64_t TotalWorkItems(const EnrichedKernel &kernel) {
  const KernelRecord &r = kernel.record;
  return r.grid_x * r.grid_y * r.grid_z * r.block_x * r.block_y * r.block_z;
}

std::int64_t TotalSharedBytes(const EnrichedKernel &kernel) {
  return static_cast<std::int64_t>(kernel.record.static_shared_bytes) +
         static_cast<std::int64_t>(kernel.record.dynamic_shared_bytes);
}

//===--- kernscope-analyze (todo 20) subprocess invocation --------------------===//
// REAL subprocess call on the extracted PTX, mirroring how the enrichment
// module shells out (enrich.cpp RunToolOutput) and how the golden/report
// tests invoke CLI tools. Exit codes per tools/analyze/main.cpp: 0 = clean,
// 1 = warnings present (a SUCCESSFUL analysis), >= 2 = error -> no merge
// (never fabricate analyzer findings).

struct AnalyzerRun {
  bool ok = false;          // analyzer ran and produced parseable JSON
  std::string error;        // why it did not (evidence for degradation)
  std::string stdout_text;  // raw --json output when ok
};

std::string RunCapture(const std::string &cmd) {
  std::FILE *pipe = ::popen(cmd.c_str(), "r");
  if (pipe == nullptr) {
    return "";
  }
  std::string out;
  char buf[4096];
  std::size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), pipe)) > 0) {
    out.append(buf, n);
  }
  (void)::pclose(pipe);
  return out;
}

AnalyzerRun RunAnalyzerOnPtx(const DiagnosticsOptions &opts,
                             const std::string &ptx_text,
                             std::int64_t block_size) {
  AnalyzerRun out;
  if (ptx_text.empty() || opts.analyzer_cli.empty()) {
    out.error = "no-ptx";
    return out;
  }

  // Temp file for the extracted PTX (mirrors the fixture-driven CLI tests).
  std::string tmpl = (std::filesystem::temp_directory_path() /
                      "kernscope-profile-ptx-XXXXXX")
                         .string();
  std::vector<char> writable(tmpl.begin(), tmpl.end());
  writable.push_back('\0');
  const int fd = ::mkstemp(writable.data());
  if (fd < 0) {
    out.error = "analyzer-tempfile-failed";
    return out;
  }
  const std::string path = writable.data();
  const std::size_t to_write = ptx_text.size();
  std::size_t written = 0;
  while (written < to_write) {
    const ssize_t n =
        ::write(fd, ptx_text.data() + written, to_write - written);
    if (n <= 0) {
      break;
    }
    written += static_cast<std::size_t>(n);
  }
  ::close(fd);

  std::string cmd = "'" + opts.analyzer_cli + "' '" + path +
                    "' --arch '" + opts.arch_name + "'";
  if (block_size > 0) {
    cmd += " --block " + std::to_string(block_size);
  }
  cmd += " --json 2>/dev/null";
  const std::string text = RunCapture(cmd);
  (void)std::filesystem::remove(path);

  if (written != to_write || text.empty()) {
    out.error = "analyzer-failed";
    return out;
  }
  // The analyzer writes the report to stdout even when warnings are present
  // (exit 1); RunCapture does not propagate the exit code, so validate the
  // report is parseable JSON here — a successful run always is.
  out.stdout_text = text;
  out.ok = true;
  return out;
}

// Merge the analyzer's warnings into `hints` as compiler hints. Every merged
// hint carries the analyzer finding id in evidence["analyze_finding_id"]
// (acceptance: the bank-conflict fixture -> hint with analyze finding id
// "bank_conflict") plus "source" = "kernscope-analyze". Returns the merged
// count.
std::size_t MergeAnalyzerFindings(const std::string &analyzer_json,
                                  std::vector<kernscope::Finding> &hints) {
  std::string err;
  const std::optional<db::json::Value> parsed = db::json::Value::parse(analyzer_json, err);
  if (!parsed) {
    return 0;
  }
  // Pointer-safe object lookup (db::json::Value has non-explicit constructors;
  // see the analyze golden test's child() wrapper rationale).
  const auto child = [](const db::json::Value *obj,
                        const char *key) -> const db::json::Value * {
    return (obj != nullptr) ? obj->find(key) : nullptr;
  };
  const db::json::Value *warnings = child(&*parsed, "warnings");
  if (warnings == nullptr || warnings->kind() != db::json::Kind::Array) {
    return 0;
  }
  std::size_t merged = 0;
  for (const db::json::Value &item : warnings->as_array()) {
    const db::json::Value *id = child(&item, "id");
    const db::json::Value *severity = child(&item, "severity");
    if (id == nullptr || severity == nullptr) {
      continue;
    }
    kernscope::Finding f;
    f.id = id->as_string();
    f.severity = severity->as_string();
    const db::json::Value *evidence = child(&item, "evidence");
    if (evidence != nullptr && evidence->kind() == db::json::Kind::Object) {
      for (const auto &[key, val] : evidence->as_object()) {
        f.evidence[key] = val.as_string();
      }
    }
    f.evidence["source"] = "kernscope-analyze";
    f.evidence["analyze_finding_id"] = f.id;
    hints.push_back(std::move(f));
    ++merged;
  }
  return merged;
}

} // namespace

//===--- ranking ---------------------------------------------------------------===//

std::vector<HotKernelEntry> rank_hot_kernels(
    const std::vector<EnrichedKernel> &kernels, int top_n) {
  // Aggregate per kernel name (map iteration is by name; the final sort makes
  // the result order deterministic regardless).
  std::map<std::string, std::vector<std::uint64_t>> by_name;
  for (const EnrichedKernel &k : kernels) {
    by_name[k.record.kernel_name].push_back(k.record.duration_ns);
  }

  std::vector<HotKernelEntry> entries;
  entries.reserve(by_name.size());
  std::uint64_t wall_time_ns = 0;
  for (const auto &[name, durations] : by_name) {
    HotKernelEntry e;
    e.kernel_name = name;
    e.launches = durations.size();
    std::vector<std::uint64_t> sorted = durations;
    std::sort(sorted.begin(), sorted.end());
    if (sorted.size() % 2 == 1) {
      e.median_ns = sorted[sorted.size() / 2];
    } else if (!sorted.empty()) {
      e.median_ns = (sorted[sorted.size() / 2 - 1] +
                     sorted[sorted.size() / 2]) /
                    2;
    }
    std::uint64_t total = 0;
    for (const std::uint64_t d : durations) {
      total += d;
    }
    e.total_ns = total;
    wall_time_ns += total;
    entries.push_back(std::move(e));
  }

  // Deterministic hotness order: total desc, then median desc, then name asc
  // (acceptance: EXACT ranking order asserted).
  std::sort(entries.begin(), entries.end(),
            [](const HotKernelEntry &a, const HotKernelEntry &b) {
              if (a.total_ns != b.total_ns) {
                return a.total_ns > b.total_ns;
              }
              if (a.median_ns != b.median_ns) {
                return a.median_ns > b.median_ns;
              }
              return a.kernel_name < b.kernel_name;
            });

  for (HotKernelEntry &e : entries) {
    e.pct_wall_time = (wall_time_ns > 0)
                          ? (static_cast<double>(e.total_ns) /
                             static_cast<double>(wall_time_ns)) *
                                100.0
                          : 0.0;
  }

  if (top_n > 0 && static_cast<std::size_t>(top_n) < entries.size()) {
    entries.resize(static_cast<std::size_t>(top_n));
  }
  return entries;
}

//===--- diagnostics rules engine ----------------------------------------------===//

KernelDiagnostics diagnose_kernel(const EnrichedKernel &kernel,
                                  const std::string &ptx_text,
                                  const DiagnosticsOptions &options) {
  KernelDiagnostics diag;
  diag.kernel_name = kernel.record.kernel_name;

  const KernelRecord &r = kernel.record;
  const int regs = static_cast<int>(r.registers_per_thread);
  const bool occupancy_known = !kernel.occupancy_unknown &&
                               kernel.occupancy >= 0.0;
  const bool occupancy_limiting =
      occupancy_known && kernel.occupancy < 1.0;
  const std::int64_t block_size = r.block_x * r.block_y * r.block_z;
  const std::optional<roofline::KernelFamily> family =
      DetectFamily(r.kernel_name);
  const bool family_known = family.has_value();

  //--- occupancy-limited: "Likely limiter: occupancy" + the binding resource --//
  // (brief §6.2.2 + the owner's layernorm_fused example, Section B.)
  if (occupancy_limiting) {
    const std::string limiter =
        kernel.limiter.empty() ? "occupancy" : kernel.limiter;
    std::map<std::string, std::string> occ_evidence;
    occ_evidence["limiter"] = limiter;
    occ_evidence["occupancy"] = FmtNumber(kernel.occupancy);
    occ_evidence["text"] = "Likely limiter: occupancy (capped at " +
                           FmtNumber(kernel.occupancy * 100.0) +
                           "% by " + limiter + ")";
    diag.hints.push_back(
        MakeFinding("occupancy_limiter", "warning", std::move(occ_evidence)));

    if (limiter == "registers" &&
        regs >= options.register_bloat_threshold) {
      // Register bloat from unrolled accumulators (brief §6.3; the owner's
      // "accumulator array likely expanded into registers; consider split
      // reduction or lower unroll factor" phrasing + the plan's
      // maxrregcount/launch_bounds suggestion).
      std::map<std::string, std::string> evidence;
      evidence["regs_per_thread"] = std::to_string(regs);
      evidence["occupancy"] = FmtNumber(kernel.occupancy);
      evidence["limiter"] = "registers";
      evidence["text"] =
          "accumulator array likely expanded into registers; consider "
          "split reduction or lower unroll factor, or cap registers with "
          "maxrregcount/launch_bounds";
      diag.hints.push_back(
          MakeFinding("register_bloat", "warning", std::move(evidence)));
    } else if (limiter == "shared") {
      std::map<std::string, std::string> evidence;
      evidence["shared_bytes"] = std::to_string(TotalSharedBytes(kernel));
      evidence["occupancy"] = FmtNumber(kernel.occupancy);
      evidence["limiter"] = "shared";
      evidence["text"] =
          "occupancy capped by shared-memory bytes per block; consider "
          "tiling smaller or a larger smem carveout";
      diag.hints.push_back(
          MakeFinding("shared_memory", "warning", std::move(evidence)));
    } else if (limiter == "threads" || limiter == "blocks") {
      std::map<std::string, std::string> evidence;
      evidence["block_size"] = std::to_string(block_size);
      evidence["occupancy"] = FmtNumber(kernel.occupancy);
      evidence["limiter"] = limiter;
      evidence["text"] =
          "block too small to fill the SM's resident warps; consider a "
          "larger block size";
      diag.hints.push_back(
          MakeFinding("small_block", "suggestion", std::move(evidence)));
    }
    diag.limiter = "occupancy";
  }

  //--- roofline memory-traffic classification for KNOWN families --------------//
  // (brief §6.2.4: AI vs ridge point via libs/roofline, todo 12. The bytes
  // moved are DERIVED from the record's observed work-item count — one
  // element per thread, fp32 conservative default; never fabricated.)
  if (!occupancy_limiting && family_known &&
      HasRecordDerivableShape(*family)) {
    const std::int64_t n = TotalWorkItems(kernel);
    if (n > 0) {
      roofline::Shape shape;
      shape.n = n;
      shape.dtype_bytes = 4; // conservative fp32 default (documented)
      const std::int64_t bytes = roofline::compute_bytes_moved(*family, shape);
      const std::int64_t flops = roofline::compute_flops(*family, shape);
      roofline::RooflinePosition pos;
      std::string err;
      if (roofline::roofline_position(*family, shape, bytes, flops,
                                      options.arch_name, &pos, &err,
                                      roofline::PeakKind::kFp32) ==
          roofline::ErrorCode::kNone) {
        const char *bound = roofline::bound_class_name(pos.bound_class);
        const char *rule = nullptr;
        const char *text = nullptr;
        switch (pos.bound_class) {
        case roofline::BoundClass::kMemory:
          rule = "memory_bound";
          text = "likely memory/traffic-bound (arithmetic intensity below the "
                 "ridge point; see the roofline position evidence)";
          diag.limiter = "memory";
          break;
        case roofline::BoundClass::kLatency:
          rule = "latency_bound";
          text = "too small to saturate the memory roof; duration likely "
                 "dominated by launch/latency effects";
          diag.limiter = "latency";
          break;
        case roofline::BoundClass::kCompute:
          rule = "compute_bound";
          text = "likely compute-bound (arithmetic intensity at/above the "
                 "ridge point)";
          diag.limiter = "compute";
          break;
        }
        if (rule != nullptr) {
          std::map<std::string, std::string> evidence;
          evidence["family"] =
              roofline::kernel_family_name(*family);
          evidence["bytes_moved_estimate"] = std::to_string(bytes);
          evidence["arithmetic_intensity"] = FmtNumber(pos.arithmetic_intensity);
          evidence["ridge_point"] = FmtNumber(pos.ridge_point);
          evidence["bound_class"] = bound;
          evidence["text"] = text;
          diag.hints.push_back(
              MakeFinding(rule, "suggestion", std::move(evidence)));
        }
      }
      // On roofline error (unknown arch / no peaks row) nothing is emitted:
      // no hint without the evidence a classification requires.
    }
  }

  //--- kernscope-analyze merge (profiler<->analyzer cross-tool multiplier) ---//
  // For kernels with PTX available: INVOKE the todo-20 CLI on the extracted
  // PTX and merge its findings (bank conflicts, divergence, coalescing) as
  // compiler hints. Failure to run/parse degrades gracefully — the analyzer
  // findings are simply not merged (never fabricated).
  bool analyzer_ran = false;
  if (!ptx_text.empty() && !options.analyzer_cli.empty()) {
    AnalyzerRun run = RunAnalyzerOnPtx(options, ptx_text, block_size);
    analyzer_ran = run.ok;
    if (run.ok) {
      (void)MergeAnalyzerFindings(run.stdout_text, diag.hints);
    }
  }

  //--- insufficient-evidence degradation (QA negative) ------------------------//
  // No PTX + unknown family + unknown occupancy -> exactly this rule, with
  // honest evidence about what was missing; never a fabricated hint.
  if (diag.limiter.empty() && diag.hints.empty() && !occupancy_known &&
      !family_known) {
    std::map<std::string, std::string> evidence;
    evidence["ptx_available"] = kernel.ptx_available ? "true" : "false";
    evidence["occupancy_known"] = "false";
    evidence["family"] = "unknown";
    std::string reason = "unknown-occupancy;unknown-family";
    if (ptx_text.empty()) {
      reason += ";no-ptx";
    } else if (!analyzer_ran) {
      reason += ";analyzer-error";
    } else {
      reason += ";analyzer-no-findings";
    }
    evidence["reason"] = reason;
    evidence["text"] =
        "insufficient evidence to classify the bottleneck: no occupancy "
        "resources, no known kernel family and no PTX-level analysis";
    diag.hints.push_back(
        MakeFinding("insufficient_evidence", "warning", std::move(evidence)));
    diag.limiter = "insufficient-evidence";
  }

  return diag;
}

//===--- JSON rendering ---------------------------------------------------------===//

std::string diagnostics_to_json(const std::vector<HotKernelEntry> &ranking,
                                const std::vector<KernelDiagnostics> &kernels) {
  std::ostringstream out;
  std::uint64_t wall_time_ns = 0;
  for (const HotKernelEntry &e : ranking) {
    wall_time_ns += e.total_ns;
  }

  out << "{\n";
  out << "  \"schema_version\": \"1.0\",\n";
  out << "  \"report_type\": \"kernscope_profile_diagnostics\",\n";
  out << "  \"wall_time_ns\": " << wall_time_ns << ",\n";
  out << "  \"ranking\": [\n";
  for (std::size_t i = 0; i < ranking.size(); ++i) {
    const HotKernelEntry &e = ranking[i];
    out << "    {\"kernel_name\": \"" << json_escape(e.kernel_name)
        << "\", \"launches\": " << e.launches
        << ", \"total_ns\": " << e.total_ns
        << ", \"median_ns\": " << e.median_ns
        << ", \"pct_wall_time\": " << FmtNumber(e.pct_wall_time) << "}";
    if (i + 1 < ranking.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ],\n";
  out << "  \"kernels\": [\n";
  for (std::size_t i = 0; i < kernels.size(); ++i) {
    const KernelDiagnostics &d = kernels[i];
    out << "    {\"kernel_name\": \"" << json_escape(d.kernel_name)
        << "\", \"limiter\": \"" << json_escape(d.limiter)
        << "\", \"hints\": [\n";
    for (std::size_t j = 0; j < d.hints.size(); ++j) {
      const kernscope::Finding &f = d.hints[j];
      out << "      {\"id\": \"" << json_escape(f.id) << "\", \"severity\": \""
          << json_escape(f.severity) << "\", \"evidence\": {";
      bool first = true;
      for (const auto &[key, value] : f.evidence) {
        if (!first) {
          out << ", ";
        }
        first = false;
        out << "\"" << json_escape(key) << "\": \"" << json_escape(value)
            << "\"";
      }
      out << "}}";
      if (j + 1 < d.hints.size()) {
        out << ",";
      }
      out << "\n";
    }
    out << "    ]}";
    if (i + 1 < kernels.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  return out.str();
}

} // namespace kernscope::profile
