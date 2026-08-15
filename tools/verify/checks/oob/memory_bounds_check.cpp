// kernscope::verify::oob — BoundsCheckPlugin (todo 31): check classes 1
// (global out-of-bounds) + 2 (shared-memory bounds).
//
// Per kernel, the plugin inspects every load/store:
//   1. resolves the access's allocation root (device global / shared array /
//      extern shared / kernel param — see oob_common.h),
//   2. computes the accessed byte span from the GEP index expressions via
//      VerifierContext::queryRange (lattice first, ScalarEvolution second —
//      research brief §4.1: SCEV bounds + propagated launch dims),
//   3. compares the span against the allocation size (statically known for
//      device/shared arrays; taken from the launch-site smem argument for
//      extern __shared__; RUNTIME and unprovable for kernel params),
//   4. emits per-access findings + one per-kernel verdict diagnostic whose
//      message carries a machine-parseable "verdict_json: {...}" payload
//      (this module's own verdict vocabulary: SAFE / MAY_OOB /
//      SYMBOLIC_DIMS — the core has no verdict enum for check classes).
//
// SOUNDNESS SCOPE (may-analysis): data-dependent indices (range unbounded by
// lattice AND SCEV) are reported MAY_OOB — never SAFE; kernel-param
// allocations get an index-bounded info note, never a static allocation
// proof; tier Symbolic (e.g. --device-only) degrades to SYMBOLIC_DIMS.
// Full scope: tools/verify/checks/oob/README.md.
//
// Registered via REGISTER_CHECK: the static registrar hands an instance to
// the global CheckRegistry — no shared file edits (todo 30 contract).

#include "oob_common.h"

#include <kernscope/verify/check_plugin.h>

#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <optional>
#include <string>
#include <vector>

namespace kernscope {
namespace verify {
namespace oob {

namespace {

using llvm::ConstantRange;
using llvm::Instruction;
using llvm::Value;

// DebugLoc-backed source location (empty when the IR carries no debug info).
SourceLoc loc_of(const Instruction* i) {
  SourceLoc loc;
  if (const llvm::DebugLoc& dl = i->getDebugLoc()) {
    loc.file = dl->getFilename().str();
    loc.line = dl->getLine();
    loc.col = dl->getColumn();
  }
  return loc;
}

// Short instruction rendering for the diagnostic.
std::string short_instruction(const Instruction& i) {
  std::string s;
  llvm::raw_string_ostream os(s);
  i.print(os);
  os.flush();
  if (s.size() > 120) {
    s.resize(120);
    s += "...";
  }
  return s;
}

// One analyzed access: everything the verdict aggregation needs.
struct AccessRecord {
  AccessFinding finding;
  int check_class = 0;
  std::string instruction;
};

// Analyzes one memory access (load/store) inside the kernel.
std::optional<AccessRecord> analyze_access(Instruction& i,
                                           VerifierContext& ctx,
                                           const SmemSizeMap& smem) {
  Value* addr = nullptr;
  bool is_store = false;
  llvm::Type* value_ty = nullptr;
  if (auto* load = llvm::dyn_cast<llvm::LoadInst>(&i)) {
    addr = load->getPointerOperand();
    value_ty = load->getType();
  } else if (auto* store = llvm::dyn_cast<llvm::StoreInst>(&i)) {
    addr = store->getPointerOperand();
    value_ty = store->getValueOperand()->getType();
    is_store = true;
  } else {
    return std::nullopt;
  }

  AccessRecord rec;
  rec.instruction = short_instruction(i);
  rec.finding.loc = loc_of(&i);
  rec.finding.is_store = is_store;

  const AllocationInfo alloc = resolveAllocation(addr, ctx.module);
  switch (alloc.root) {
    case AllocationInfo::Root::DeviceGlobal:
      rec.check_class = CheckClass::kGlobalOob;
      break;
    case AllocationInfo::Root::SharedArray:
    case AllocationInfo::Root::SharedDynamic:
      rec.check_class = CheckClass::kSmemBounds;
      break;
    case AllocationInfo::Root::KernelParam:
      rec.check_class = CheckClass::kGlobalOob;
      break;
    case AllocationInfo::Root::Local:
    case AllocationInfo::Root::Unknown:
      return std::nullopt;  // out of this module's v1 scope (documented).
  }
  rec.finding.alloc = alloc.name;
  rec.finding.check_class = rec.check_class;

  // Byte-span of the access: GEP index-expression ranges via queryRange
  // (lattice -> SCEV).  The span holds possible element START offsets, so
  // the in-bounds check adds the access width for the last accessed byte.
  std::string index_range;
  uint64_t innermost_stride = 0;
  std::optional<ConstantRange> span =
      gepByteSpan(addr, ctx, index_range, innermost_stride);
  const bool gep_chain = isGepChain(addr);
  const uint64_t width =
      gep_chain ? innermost_stride : accessByteWidth(value_ty, ctx.module);
  if (gep_chain && !span) {
    // A GEP whose index range is unbounded (data-dependent / runtime index).
    rec.finding.kind = AccessFinding::Kind::Unbounded;
    rec.finding.detail = "index expression cannot be bounded (lattice/SCEV): "
                         "may-analysis — no in-bounds claim";
    return rec;
  }
  if (!gep_chain) {
    span = ConstantRange::getNonEmpty(llvm::APInt(64, 0), llvm::APInt(64, 1));
  }
  rec.finding.span = range_render(*span);
  rec.finding.index_range = index_range;

  if (alloc.root == AllocationInfo::Root::SharedDynamic) {
    // extern __shared__: the size comes from the launch-site smem argument.
    const auto it = smem.find(ctx.kernel.function);
    if (it == smem.end()) {
      rec.finding.kind = AccessFinding::Kind::Unbounded;
      rec.finding.detail = "dynamic shared memory size unknown (no constant "
                           "launch-site smem argument): may-analysis — no "
                           "in-bounds claim";
      return rec;
    }
    const std::optional<std::uint64_t> &launch_smem = it->second;
    if (!launch_smem.has_value()) {
      rec.finding.kind = AccessFinding::Kind::Unbounded;
      rec.finding.detail = "dynamic shared memory size unknown (no constant "
                           "launch-site smem argument): may-analysis — no "
                           "in-bounds claim";
      return rec;
    }
    rec.finding.alloc_known = true;
    rec.finding.bound_bytes = *launch_smem;
    rec.finding.detail = "dynamic shared memory: bound from launch-site "
                         "smem argument (" +
                         std::to_string(*launch_smem) + " bytes)";
  } else if (alloc.known_size) {
    rec.finding.alloc_known = true;
    rec.finding.bound_bytes = alloc.size_bytes;
  }

  if (!rec.finding.alloc_known) {
    // Kernel parameter: runtime allocation.  The INDEX is proven bounded
    // (that is the class-1 contract: index expressions bounded via SCEV +
    // launch dims); a static allocation proof is impossible.
    rec.finding.kind = AccessFinding::Kind::ParamBounded;
    rec.finding.detail = "allocation size is a runtime kernel parameter: "
                         "index proven bounded, static allocation proof not "
                         "possible (may-analysis scope)";
    return rec;
  }

  // Known-size allocation: in-bounds iff the last accessed byte stays
  // inside the allocation.  The span's upper bound is (max start offset + 1);
  // the element width w covers bytes [start, start + w), so the condition is
  // (upper - 1) + w <= size.
  llvm::APInt last = span->getUpper();
  last -= 1;
  last += width;
  const bool in_bounds = last.ule(rec.finding.bound_bytes);
  rec.finding.kind = in_bounds ? AccessFinding::Kind::InBounds
                               : AccessFinding::Kind::MayOob;
  rec.finding.detail = in_bounds
                           ? std::string("access within allocation ") +
                                 alloc.name + " (" +
                                 std::to_string(rec.finding.bound_bytes) +
                                 " bytes)"
                           : std::string("access may exceed allocation ") +
                                 alloc.name + " (" +
                                 std::to_string(rec.finding.bound_bytes) +
                                 " bytes)";
  return rec;
}

const char* kind_name(AccessFinding::Kind k) {
  switch (k) {
    case AccessFinding::Kind::InBounds:
      return "IN_BOUNDS";
    case AccessFinding::Kind::MayOob:
      return "MAY_OOB";
    case AccessFinding::Kind::Unbounded:
      return "UNBOUNDED";
    case AccessFinding::Kind::ParamBounded:
      return "PARAM_BOUNDED";
  }
  return "?";
}

// JSON rendering of one access finding (module-owned format; the corpus test
// parses "verdict_json:" payloads with its own mini parser).
std::string finding_json(const AccessFinding& f) {
  std::string out = "{\"op\":\"";
  out += f.is_store ? "store" : "load";
  out += "\",\"line\":" + std::to_string(f.loc.line);
  out += ",\"alloc\":\"" + jsonEscape(f.alloc) + "\"";
  out += ",\"known\":" + std::string(f.alloc_known ? "true" : "false");
  if (f.alloc_known) {
    out += ",\"bound\":" + std::to_string(f.bound_bytes);
  }
  out += ",\"range\":\"" + jsonEscape(f.index_range) + "\"";
  out += ",\"span\":\"" + jsonEscape(f.span) + "\"";
  out += ",\"kind\":\"" + std::string(kind_name(f.kind)) + "\"";
  out += ",\"note\":\"" + jsonEscape(f.detail) + "\"}";
  return out;
}

}  // namespace

class BoundsCheckPlugin : public CheckPlugin {
 public:
  BoundsCheckPlugin() {
    id = "global-oob";
    name = "Global OOB + shared-memory bounds";
    check_class_ids = {CheckClass::kGlobalOob, CheckClass::kSmemBounds};
  }

  std::vector<Diagnostic> run(VerifierContext& ctx) const override {
    std::vector<Diagnostic> out;
    const std::string& fn = ctx.kernel.name;
    // Scanned per run: the scan is module-global and cheap (the core's
    // resolveLaunchConfig does the same per kernel); a thread_local cache
    // keyed by Module* would go stale when allocators reuse the pointer of
    // a freed module across analyses in one process.
    const SmemSizeMap& smem = scanLaunchSmem(ctx.module);

    // Collect the kernel's memory accesses (budgeted: one slot per access).
    std::vector<AccessRecord> accesses;
    for (llvm::BasicBlock& bb : *ctx.kernel.function) {
      for (Instruction& i : bb) {
        if (!ctx.diagnostics.budget().acquire(fn, 1)) {
          break;  // budget latched; engine emits the degraded verdict
        }
        auto rec = analyze_access(i, ctx, smem);
        if (rec) {
          accesses.push_back(std::move(*rec));
        }
      }
    }
    if (accesses.empty()) {
      return out;  // no memory accesses to bound for this kernel
    }

    // Per-class views.
    auto in_class = [&](int cls) {
      std::vector<const AccessRecord*> v;
      for (const AccessRecord& a : accesses) {
        if (a.check_class == cls) {
          v.push_back(&a);
        }
      }
      return v;
    };
    const std::vector<const AccessRecord*> global =
        in_class(CheckClass::kGlobalOob);
    const std::vector<const AccessRecord*> smem_accesses =
        in_class(CheckClass::kSmemBounds);
    const std::vector<const AccessRecord*> groups[] = {global, smem_accesses};

    if (ctx.launchConfig().symbolic()) {
      // Degraded verdict: no launch config -> no bounds can be proven
      // (soundness over false safety; the QA negative asserts this path).
      for (const std::vector<const AccessRecord*>& group : groups) {
        if (group.empty()) {
          continue;
        }
        Diagnostic d;
        d.severity = Severity::Warning;
        d.class_id = group.front()->check_class;
        d.class_name = check_class_name(d.class_id);
        d.function = fn;
        d.loc = group.front()->finding.loc;
        d.message =
            "verdict degraded: launch configuration is not a compile-time "
            "constant (or the module was compiled --device-only) — no "
            "memory-access bound can be proven; "
            "verdict_json: {\"check\":" + std::to_string(d.class_id) +
            ",\"kernel\":\"" + fn + "\",\"verdict\":\"" + kVerdictSymbolicDims +
            "\",\"tier\":\"Symbolic\",\"accesses\":" +
            std::to_string(group.size()) +
            ",\"reason\":\"launch dims unknown (device-only ingestion or "
            "runtime dims): bounds unprovable — runtime check required\"}";
        out.push_back(std::move(d));
      }
      return out;
    }

    // Emit per-class verdicts.
    for (const std::vector<const AccessRecord*>& group : groups) {
      if (group.empty()) {
        continue;
      }
      const int cls = group.front()->check_class;
      size_t may_oob = 0;
      std::string findings = "[";
      for (size_t k = 0; k < group.size(); ++k) {
        const AccessFinding& f = group[k]->finding;
        if (k) {
          findings += ",";
        }
        findings += finding_json(f);
        if (f.kind == AccessFinding::Kind::MayOob ||
            f.kind == AccessFinding::Kind::Unbounded) {
          ++may_oob;
        }
      }
      findings += "]";

      const bool safe = may_oob == 0;
      Diagnostic d;
      d.severity = safe ? Severity::Info : Severity::Warning;
      d.class_id = cls;
      d.class_name = check_class_name(cls);
      d.function = fn;
      d.loc = group.front()->finding.loc;
      const char* tier = "Exact";
      if (ctx.launchConfig().bounded()) {
        tier = "UpperBound";
      }
      const std::string verdict = safe ? kVerdictSafe : kVerdictMayOob;
      const std::string summary = cls == CheckClass::kGlobalOob
                                      ? "global OOB check"
                                      : "shared-memory bounds check";
      std::string msg = summary + ": " + std::to_string(group.size());
      msg += " access(es), ";
      msg += std::to_string(may_oob);
      msg += " possibly out-of-bounds; verdict_json: {\"check\":";
      msg += std::to_string(cls);
      msg += ",\"kernel\":\"";
      msg += fn;
      msg += "\",\"verdict\":\"";
      msg += verdict;
      msg += "\",\"tier\":\"";
      msg += tier;
      msg += "\",\"accesses\":";
      msg += std::to_string(group.size());
      msg += ",\"may_oob\":";
      msg += std::to_string(may_oob);
      msg += ",\"findings\":";
      msg += findings;
      msg += "}";
      d.message = std::move(msg);
      out.push_back(std::move(d));
    }
    return out;
  }
};

// NOLINTNEXTLINE(misc-use-anonymous-namespace)  // macro-expanded plugin registry entry (must be namespace-scope static)
REGISTER_CHECK(BoundsCheckPlugin);

}  // namespace oob
}  // namespace verify
}  // namespace kernscope
