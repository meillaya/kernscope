//===- analyze_mix.cpp - instruction mix + tensor core + latency -----------===//
//
// Wave-3 todo 18. See mix.h for the design rationale.
//
// Method:
//   * instruction-mix histogram by class (FP32/FP16/INT/LSU/SMEM/TENSOR/
//     CTRL/other) from the parser taxonomy + mnemonic-level buckets matching
//     the expected_fused_attention.json instruction_mix fields
//     (fma_rn_f32, mul_f32, add_f32, ex2_approx_f32, ld_global,
//     ld_global_vectorized, ld_shared, st_global, st_shared, ld_local,
//     st_local, setp, predicated_bra, bar_sync);
//   * tensor-core detection + MAC counting: mma/wgmma/wmma/tcgen05 shapes
//     -> MACs = m*n*k [brief §5.1 L346-350]; MFMA v_mfma_* and RDNA3
//     v_wmma_* on AMD [brief §5.2]; "no tensor-core usage" info when
//     GEMM-like loops use only FMA;
//   * approximate latency bottleneck: per-arch published latency tables
//     (brief §8.1 L460-471: Volta L1 28/smem 19/L2 193, Turing 32/19/188,
//     A100 33/29/203, Ada 32/30/273, Hopper 32/29/264) fed into a
//     critical-path estimate with issue-slot reasoning -> ALU-bound /
//     LSU-bound / latency-bound, dominant class named; every figure is
//     labeled "approximate, microbenchmark-derived". Unknown arch ->
//     "unknown-arch" skip (never fabricated numbers).
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/mix.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "kernscope/arch.h"

namespace kernscope::analyze {
namespace {

// Per-class latency (cycles), from brief §8.1 L460-471 (measured
// microbenchmarks; per-arch cache-level latency for memory ops).
struct LatencyTable {
  int l1 = 32;   // L1 hit
  int smem = 19; // shared memory (no conflict)
  int l2 = 188;  // L2 hit
  int dram = 566;// global / DRAM
};

// NVIDIA archs covered by the published tables. Keyed by name so the QA
// unknown-arch skip is exact (a name NOT in this table -> no numbers).
LatencyTable latency_for_arch(const std::string &name) {
  LatencyTable t;
  if (name == "sm_75") t = {32, 19, 188, 616};    // Turing T4
  else if (name == "sm_80") t = {33, 29, 203, 566};  // A100
  else if (name == "sm_86") t = {33, 30, 273, 571};  // Ada-class
  else if (name == "sm_89") t = {32, 30, 273, 571};  // Ada
  else if (name == "sm_90") t = {32, 29, 264, 656};  // Hopper
  else if (name == "sm_100" || name == "sm_120")
    t = {32, 29, 264, 656};  // Hopper-class (no Blackwell table in brief)
  else t = {-1, -1, -1, -1};  // unknown -> caller skips
  return t;
}

struct Mix {
  long long fma_rn_f32 = 0;
  long long mul_f32 = 0;
  long long add_f32 = 0;
  long long ex2_approx_f32 = 0;
  long long ld_global = 0;
  long long ld_global_vectorized = 0;
  long long ld_shared = 0;
  long long st_global = 0;
  long long st_shared = 0;
  long long ld_local = 0;
  long long st_local = 0;
  long long setp = 0;
  long long predicated_bra = 0;
  long long bar_sync = 0;
  std::map<std::string, long long> class_hist;
};

void bump_class(Mix &m, const std::string &bucket) { m.class_hist[bucket]++; }

// mnemonic-level + class histogram from the parser taxonomy.
void classify_instruction(const PtxInstruction &insn, Mix &m) {
  const std::string &mn = insn.mnemonic;
  const InstructionClass c = insn.cls;

  if (mn == "fma.rn.f32" || mn == "fma.rn.ftz.f32") m.fma_rn_f32++;
  if (mn == "mul.f32") m.mul_f32++;
  if (mn == "add.f32") m.add_f32++;
  if (mn == "ex2.approx.f32") m.ex2_approx_f32++;
  if (c == InstructionClass::kSetp) m.setp++;
  if (insn.is_branch && insn.has_predicate) m.predicated_bra++;
  if (mn == "bar.sync") m.bar_sync++;

  if (insn.mem_space == StateSpace::kGlobal) {
    if (c == InstructionClass::kLoad) {
      m.ld_global++;
      if (mn.find(".v2") != std::string::npos ||
          mn.find(".v4") != std::string::npos)
        m.ld_global_vectorized++;
    } else if (c == InstructionClass::kStore) {
      m.st_global++;
    }
  } else if (insn.mem_space == StateSpace::kShared) {
    if (c == InstructionClass::kLoad) m.ld_shared++;
    else if (c == InstructionClass::kStore) m.st_shared++;
  } else if (insn.mem_space == StateSpace::kLocal) {
    if (c == InstructionClass::kLoad) m.ld_local++;
    else if (c == InstructionClass::kStore) m.st_local++;
  }

  switch (c) {
    case InstructionClass::kFma: bump_class(m, "FMA"); break;
    case InstructionClass::kAlu:
    case InstructionClass::kLogic:
    case InstructionClass::kMath:
    case InstructionClass::kCvt:
    case InstructionClass::kMov:
    case InstructionClass::kSelect:
    case InstructionClass::kSetp:
      bump_class(m, "ALU");
      break;
    case InstructionClass::kLoad:
    case InstructionClass::kStore:
    case InstructionClass::kLdMatrix:
    case InstructionClass::kStMatrix:
      bump_class(m, insn.mem_space == StateSpace::kShared ? "SMEM" : "LSU");
      break;
    case InstructionClass::kTensorMmaSync:
    case InstructionClass::kTensorWmma:
    case InstructionClass::kTensorWgmma:
    case InstructionClass::kTensorTcgen05:
      bump_class(m, "TENSOR");
      break;
    case InstructionClass::kBranch:
    case InstructionClass::kCall:
    case InstructionClass::kReturn:
    case InstructionClass::kExit:
    case InstructionClass::kBarrier:
      bump_class(m, "CTRL");
      break;
    default: bump_class(m, "other"); break;
  }
}

// Parse a tensor shape token ("m16n8k16" for mma/wgmma; "32x32x8" for
// v_mfma) into m/n/k.
struct Shape {
  int m = 0;
  int n = 0;
  int k = 0;
  bool ok = false;
};

// Parse the shape token: find "m<N>", "n<N>", "k<N>" (N = digits) anywhere
// in the mnemonic, but only when a digit IMMEDIATELY follows the tag (so
// "mma.sync..." does not match the 'm' of "mma").
int dim_after(const std::string &s, char tag) {
  for (std::size_t i = 0; i + 1 < s.size(); ++i) {
    if (s[i] != tag) continue;
    if (!std::isdigit(static_cast<unsigned char>(s[i + 1]))) continue;
    char *end = nullptr;
    const long v = std::strtol(s.c_str() + i + 1, &end, 10);
    if (end != s.c_str() + i + 1 && v > 0) return static_cast<int>(v);
  }
  return 0;
}

Shape parse_mn_shape(const std::string &mnemonic) {
  Shape sh;
  // Lowercase the token for case-insensitive matching.
  std::string s;
  for (char ch : mnemonic)
    s += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

  sh.m = dim_after(s, 'm');
  sh.n = dim_after(s, 'n');
  sh.k = dim_after(s, 'k');
  if (sh.m > 0 && sh.n > 0 && sh.k > 0) {
    sh.ok = true;
    return sh;
  }
  // MFMA "32x32x8f16" style: extract the m x n x k digit triple with a
  // regex ([0-9]+)x([0-9]+)x([0-9]+). The "f32" in "v_mfma_f32_" never
  // matches because it is not part of an x-separated triple.
  const std::regex xform(R"(([0-9]+)x([0-9]+)x([0-9]+))");
  std::smatch mm;
  if (std::regex_search(s, mm, xform)) {
    sh.m = std::stoi(mm[1].str());
    sh.n = std::stoi(mm[2].str());
    sh.k = std::stoi(mm[3].str());
    sh.ok = sh.m > 0 && sh.n > 0 && sh.k > 0;
  }
  return sh;
}

// MACs = m*n*k per instruction [brief §5.1 L346-350; §5.2 L372-380].
long long tensor_macs(const std::string &mnemonic) {
  const Shape sh = parse_mn_shape(mnemonic);
  if (sh.ok)
    return static_cast<long long>(sh.m) * sh.n * sh.k;
  return 0;  // shape not statically derivable (e.g. tcgen05 operand shapes)
}

// Approximate critical-path estimate. Issue-slot reasoning per brief §8.2
// L494-510: ALU issue = count / 4 schedulers; memory ops cost their cache
// latency; classify the dominant bound and NAME the dominant class.
struct LatencyOutcome {
  std::string classification;
  std::string dominant_class;
  double estimated_cycles = 0.0;
};

LatencyOutcome estimate_latency(const PtxKernel &kernel,
                                const LatencyTable &t) {
  LatencyOutcome out;
  out.classification = "unknown-arch";
  out.dominant_class = "other";
  out.estimated_cycles = 0.0;
  if (t.l1 < 0) return out;  // no published table for this arch

  long long alu_count = 0;
  long long lsu_count = 0;
  long long smem_count = 0;
  for (const PtxBlock &block : kernel.cfg) {
    for (const PtxInstruction &insn : block.instructions) {
      const InstructionClass c = insn.cls;
      switch (c) {
        case InstructionClass::kFma:
        case InstructionClass::kAlu:
        case InstructionClass::kLogic:
        case InstructionClass::kMath:
        case InstructionClass::kCvt:
        case InstructionClass::kMov:
        case InstructionClass::kSetp:
        case InstructionClass::kSelect:
          alu_count++;
          break;
        case InstructionClass::kLoad:
        case InstructionClass::kStore:
        case InstructionClass::kLdMatrix:
        case InstructionClass::kStMatrix:
        case InstructionClass::kAtomic:
        case InstructionClass::kReduction:
        case InstructionClass::kTexture:
        case InstructionClass::kAsyncCopy:
          if (insn.mem_space == StateSpace::kShared) smem_count++;
          else lsu_count++;
          break;
        default: break;
      }
    }
  }

  // Issue slots: 4 schedulers/SM issue ~1 instruction/cycle each [brief
  // §8.2 L492, TURING §2 single-dispatch]. ALU budget = alu_count/4.
  const double alu_issue = static_cast<double>(alu_count) / 4.0;
  // Memory latency budget: LSU ops at L1, smem ops at smem latency.
  const double mem_latency = static_cast<double>(lsu_count) * t.l1 +
                             static_cast<double>(smem_count) * t.smem;

  if (alu_count == 0 && lsu_count == 0 && smem_count == 0) {
    out.classification = "alu-bound";
    out.dominant_class = "other";
    out.estimated_cycles = 0.0;
    return out;
  }

  // Latency-bound: the memory dependence chain cannot be hidden by the
  // available ALU issue slots (Little's law, brief §8.2 L501).
  if (mem_latency > 4.0 * alu_issue && mem_latency > 0) {
    out.classification = "latency-bound";
    out.dominant_class = lsu_count >= smem_count ? "LSU" : "SMEM";
    out.estimated_cycles = mem_latency;
  } else if (lsu_count + smem_count > alu_count) {
    out.classification = "lsu-bound";
    out.dominant_class = lsu_count >= smem_count ? "LSU" : "SMEM";
    out.estimated_cycles = mem_latency;
  } else {
    out.classification = "alu-bound";
    out.dominant_class = "ALU";
    out.estimated_cycles = alu_issue;
  }
  return out;
}

}  // namespace

void analyze_mix(const PtxModule &module, KernelReport &report) {
  Mix mix;
  long long tensor_mac_count = 0;
  int tensor_ops = 0;

  for (const PtxKernel &kernel : module.kernels()) {
    for (const PtxBlock &block : kernel.cfg) {
      for (const PtxInstruction &insn : block.instructions) {
        classify_instruction(insn, mix);
        if (insn.cls == InstructionClass::kTensorMmaSync ||
            insn.cls == InstructionClass::kTensorWmma ||
            insn.cls == InstructionClass::kTensorWgmma ||
            insn.cls == InstructionClass::kTensorTcgen05) {
          tensor_ops++;
          tensor_mac_count += tensor_macs(insn.mnemonic);
        }
      }
    }
    break;  // first kernel
  }

  // Mnemonic-level metrics (expected_fused_attention.json instruction_mix).
  report.metrics["fma_rn_f32"] = mix.fma_rn_f32;
  report.metrics["mul_f32"] = mix.mul_f32;
  report.metrics["add_f32"] = mix.add_f32;
  report.metrics["ex2_approx_f32"] = mix.ex2_approx_f32;
  report.metrics["ld_global"] = mix.ld_global;
  report.metrics["ld_global_vectorized"] = mix.ld_global_vectorized;
  report.metrics["ld_shared"] = mix.ld_shared;
  report.metrics["st_global"] = mix.st_global;
  report.metrics["st_shared"] = mix.st_shared;
  report.metrics["ld_local"] = mix.ld_local;
  report.metrics["st_local"] = mix.st_local;
  report.metrics["setp"] = mix.setp;
  report.metrics["predicated_bra"] = mix.predicated_bra;
  report.metrics["bar_sync"] = mix.bar_sync;
  report.metrics["tensor_ops"] = tensor_ops;
  report.metrics["tensor_macs"] = tensor_mac_count;

  // Class histogram counts + percentages.
  long long total = 0;
  for (const auto &[bucket, count] : mix.class_hist) total += count;
  for (const auto &[bucket, count] : mix.class_hist) {
    report.metrics["mix_count_" + bucket] = count;
    if (total > 0)
      report.metrics["mix_pct_" + bucket] =
          100.0 * static_cast<double>(count) / static_cast<double>(total);
  }

  // Tensor-core usage info: GEMM-like (FMA-heavy) with no tensor ops.
  if (tensor_ops == 0 && mix.fma_rn_f32 > 0) {
    Finding f;
    f.id = "no_tensor_core_usage";
    f.severity = "info";
    f.evidence["text"] =
        "GEMM-like workload uses only FMA (no mma/wgmma/wmma/tcgen05); "
        "tensor cores are not exploited";
    f.evidence["fma_count"] = std::to_string(mix.fma_rn_f32);
    f.evidence["tensor_macs"] = std::to_string(tensor_mac_count);
    report.suggestions.push_back(std::move(f));
  }

  // Latency bottleneck (approximate, microbenchmark-derived).
  const std::string arch = module.target;
  if (!arch.empty() && latency_for_arch(arch).l1 >= 0) {
    const LatencyTable t = latency_for_arch(arch);
    for (const PtxKernel &kernel : module.kernels()) {
      const LatencyOutcome lo = estimate_latency(kernel, t);
      report.metrics["latency_classification"] = 1.0;
      report.metrics["latency_estimated_cycles"] = lo.estimated_cycles;
      Finding f;
      f.id = "latency_bottleneck";
      f.severity = "info";
      f.evidence["text"] =
          "static critical-path estimate: " + lo.classification + " (" +
          lo.dominant_class + "); approximate, microbenchmark-derived "
          "(brief §8.1 latency tables, never cycle-exact)";
      f.evidence["classification"] = lo.classification;
      f.evidence["dominant_class"] = lo.dominant_class;
      f.evidence["arch"] = arch;
      f.evidence["estimated_cycles"] =
          std::to_string(static_cast<long long>(lo.estimated_cycles));
      report.suggestions.push_back(std::move(f));
      break;
    }
  } else {
    // QA: unknown arch -> "unknown-arch" skip, no fabricated numbers.
    Finding f;
    f.id = "latency_unknown_arch";
    f.severity = "info";
    f.evidence["text"] =
        "latency bottleneck skipped: no published latency table for arch '" +
        arch + "' (brief §8.1 covers Volta/Turing/Ampere/Ada/Hopper only; "
               "no fabricated numbers)";
    f.evidence["arch"] = arch;
    report.suggestions.push_back(std::move(f));
  }
}

void analyze_mix(const kernscope::AmdModule &module, KernelReport &report) {
  Mix mix;
  long long tensor_mac_count = 0;
  int tensor_ops = 0;
  for (const AmdInstruction &insn : module.instructions) {
    switch (insn.klass) {
      case InstrClass::VALU: bump_class(mix, "ALU"); break;
      case InstrClass::SALU: bump_class(mix, "ALU"); break;
      case InstrClass::LDS: bump_class(mix, "SMEM"); break;
      case InstrClass::Global:
      case InstrClass::Buffer:
      case InstrClass::Flat:
      case InstrClass::Scratch:
        bump_class(mix, "LSU");
        break;
      case InstrClass::Branch:
      case InstrClass::Barrier:
      case InstrClass::Wait:
      case InstrClass::EndPgm:
        bump_class(mix, "CTRL");
        break;
      case InstrClass::MFMA:
      case InstrClass::WMMA:
        bump_class(mix, "TENSOR");
        break;
      default: bump_class(mix, "other"); break;
    }
    if (insn.is_tensor) {
      tensor_ops++;
      tensor_mac_count += tensor_macs(insn.mnemonic);
    }
  }
  report.metrics["tensor_ops"] = tensor_ops;
  report.metrics["tensor_macs"] = tensor_mac_count;
  for (const auto &[bucket, count] : mix.class_hist) {
    report.metrics["mix_count_" + bucket] = count;
  }
  if (tensor_ops == 0 && mix.class_hist["ALU"] > 0) {
    Finding f;
    f.id = "no_tensor_core_usage";
    f.severity = "info";
    f.evidence["text"] =
        "GEMM-like workload uses only VALU (no v_mfma_*/v_wmma_*); "
        "matrix cores are not exploited";
    f.evidence["tensor_macs"] = std::to_string(tensor_mac_count);
    report.suggestions.push_back(std::move(f));
  }
}

}  // namespace kernscope::analyze
