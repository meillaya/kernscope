#include <kernscope/verify/verifier_state.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>

#include <kernscope/verify/check_plugin.h>
#include <kernscope/verify/diagnostic.h>
#include <kernscope/verify/launch_config.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace kernscope {
namespace verify {

using llvm::BasicBlock;
using llvm::ConstantInt;
using llvm::ConstantRange;
using llvm::DenseMap;
using llvm::Function;
using llvm::GetElementPtrInst;
using llvm::Instruction;
using llvm::Module;
using llvm::Value;

namespace {

const char* lane_name(Lane lane) {
  switch (lane) {
    case Lane::ThreadX:
      return "threadIdx.x";
    case Lane::ThreadY:
      return "threadIdx.y";
    case Lane::ThreadZ:
      return "threadIdx.z";
    case Lane::BlockDimX:
      return "blockDim.x";
    case Lane::BlockDimY:
      return "blockDim.y";
    case Lane::BlockDimZ:
      return "blockDim.z";
    case Lane::BlockX:
      return "blockIdx.x";
    case Lane::BlockY:
      return "blockIdx.y";
    case Lane::BlockZ:
      return "blockIdx.z";
    case Lane::GridDimX:
      return "gridDim.x";
    case Lane::GridDimY:
      return "gridDim.y";
    case Lane::GridDimZ:
      return "gridDim.z";
  }
  return "?";
}

// Maps an nvptx/amdgcn coordinate intrinsic name to its Lane.
std::optional<Lane> lane_for_intrinsic(llvm::StringRef name) {
  static const llvm::StringMap<Lane> table = {
      {"llvm.nvvm.read.ptx.sreg.tid.x", Lane::ThreadX},
      {"llvm.nvvm.read.ptx.sreg.tid.y", Lane::ThreadY},
      {"llvm.nvvm.read.ptx.sreg.tid.z", Lane::ThreadZ},
      {"llvm.nvvm.read.ptx.sreg.ntid.x", Lane::BlockDimX},
      {"llvm.nvvm.read.ptx.sreg.ntid.y", Lane::BlockDimY},
      {"llvm.nvvm.read.ptx.sreg.ntid.z", Lane::BlockDimZ},
      {"llvm.nvvm.read.ptx.sreg.ctaid.x", Lane::BlockX},
      {"llvm.nvvm.read.ptx.sreg.ctaid.y", Lane::BlockY},
      {"llvm.nvvm.read.ptx.sreg.ctaid.z", Lane::BlockZ},
      {"llvm.nvvm.read.ptx.sreg.nctaid.x", Lane::GridDimX},
      {"llvm.nvvm.read.ptx.sreg.nctaid.y", Lane::GridDimY},
      {"llvm.nvvm.read.ptx.sreg.nctaid.z", Lane::GridDimZ},
      {"llvm.amdgcn.workitem.id.x", Lane::ThreadX},
      {"llvm.amdgcn.workitem.id.y", Lane::ThreadY},
      {"llvm.amdgcn.workitem.id.z", Lane::ThreadZ},
      {"llvm.amdgcn.workgroup.size.x", Lane::BlockDimX},
      {"llvm.amdgcn.workgroup.size.y", Lane::BlockDimY},
      {"llvm.amdgcn.workgroup.size.z", Lane::BlockDimZ},
      {"llvm.amdgcn.workgroup.id.x", Lane::BlockX},
      {"llvm.amdgcn.workgroup.id.y", Lane::BlockY},
      {"llvm.amdgcn.workgroup.id.z", Lane::BlockZ},
      {"llvm.amdgcn.grid.size.x", Lane::GridDimX},
      {"llvm.amdgcn.grid.size.y", Lane::GridDimY},
      {"llvm.amdgcn.grid.size.z", Lane::GridDimZ},
  };
  auto it = table.find(name);
  if (it == table.end()) {
    return std::nullopt;
  }
  return it->second;
}

// Per-function analysis cache (DominatorTree/LoopInfo/AssumptionCache/
// TargetLibraryInfo feed ScalarEvolution, research brief §4.1).
struct FuncAnalysis {
  FuncAnalysis(Function& f)
      : tlii(llvm::Triple(f.getParent()->getTargetTriple())),
        tli(tlii),
        ac(f),
        dt(f),
        li(dt),
        se(f, tli, ac, dt, li) {}
  llvm::TargetLibraryInfoImpl tlii;
  llvm::TargetLibraryInfo tli;
  llvm::AssumptionCache ac;
  llvm::DominatorTree dt;
  llvm::LoopInfo li;
  llvm::ScalarEvolution se;
};

// Range lattice for one kernel: value -> sound unsigned range (value
// bitwidth for scalars, 64-bit offset range for GEPs). Only widens.
struct Lattice {
  DenseMap<const Value*, ConstantRange> ranges;
  size_t bounded = 0;
};

}  // namespace

bool KernelInfo::rangeBounded(Lane lane) const {
  const ConstantRange& r = range(lane);
  return !r.isFullSet() && !r.isEmptySet();
}

// --------------------------------------------------------------------------
// DiagnosticEngine
// --------------------------------------------------------------------------

void DiagnosticEngine::emit(Diagnostic d) {
  diagnostics_.push_back(std::move(d));
}

void DiagnosticEngine::emitBudgetExceeded(const std::string& function) {
  if (std::find(budget_latched_functions_.begin(), budget_latched_functions_.end(),
                function) != budget_latched_functions_.end()) {
    return;
  }
  budget_latched_functions_.push_back(function);
  Diagnostic d;
  d.severity = Severity::Warning;
  d.class_id = EngineClass::kBudgetExceeded;
  d.function = function;
  d.message = "budget exceeded: explored-state cap (" +
              std::to_string(budget_.max_states_per_function()) +
              " states/function) reached in " + function +
              "; analysis degraded (terminated, not hung)";
  emit(std::move(d));
}

// --------------------------------------------------------------------------
// VerifierState impl
// --------------------------------------------------------------------------

struct VerifierState::Impl {
  std::unordered_map<const Function*, std::unique_ptr<FuncAnalysis>> analyses;
  // Per-kernel lattice (parallel to kernels_).
  std::vector<Lattice> lattices;
  std::vector<size_t> bounded_access_counts;
  uint64_t total_bounded_accesses = 0;
};

namespace {

// Range for the given launch config: intrinsic lane -> unsigned range.
std::vector<ConstantRange> compute_intrinsic_ranges(const LaunchConfig& cfg) {
  constexpr unsigned W = 32;
  const ConstantRange full = ConstantRange::getFull(W);
  std::vector<ConstantRange> lanes(12, full);
  if (cfg.tier == LaunchTier::Exact) {
    lanes[static_cast<size_t>(Lane::ThreadX)] = llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 0), llvm::APInt(32, cfg.block.x));
    lanes[static_cast<size_t>(Lane::ThreadY)] = llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 0), llvm::APInt(32, cfg.block.y));
    lanes[static_cast<size_t>(Lane::ThreadZ)] = llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 0), llvm::APInt(32, cfg.block.z));
    lanes[static_cast<size_t>(Lane::BlockDimX)] =
        llvm::ConstantRange::getNonEmpty(llvm::APInt(32, cfg.block.x), llvm::APInt(32, cfg.block.x + 1));
    lanes[static_cast<size_t>(Lane::BlockDimY)] =
        llvm::ConstantRange::getNonEmpty(llvm::APInt(32, cfg.block.y), llvm::APInt(32, cfg.block.y + 1));
    lanes[static_cast<size_t>(Lane::BlockDimZ)] =
        llvm::ConstantRange::getNonEmpty(llvm::APInt(32, cfg.block.z), llvm::APInt(32, cfg.block.z + 1));
    lanes[static_cast<size_t>(Lane::BlockX)] = llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 0), llvm::APInt(32, cfg.grid.x));
    lanes[static_cast<size_t>(Lane::BlockY)] = llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 0), llvm::APInt(32, cfg.grid.y));
    lanes[static_cast<size_t>(Lane::BlockZ)] = llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 0), llvm::APInt(32, cfg.grid.z));
    lanes[static_cast<size_t>(Lane::GridDimX)] =
        llvm::ConstantRange::getNonEmpty(llvm::APInt(32, cfg.grid.x), llvm::APInt(32, cfg.grid.x + 1));
    lanes[static_cast<size_t>(Lane::GridDimY)] =
        llvm::ConstantRange::getNonEmpty(llvm::APInt(32, cfg.grid.y), llvm::APInt(32, cfg.grid.y + 1));
    lanes[static_cast<size_t>(Lane::GridDimZ)] =
        llvm::ConstantRange::getNonEmpty(llvm::APInt(32, cfg.grid.z), llvm::APInt(32, cfg.grid.z + 1));
  } else if (cfg.tier == LaunchTier::UpperBound) {
    lanes[static_cast<size_t>(Lane::ThreadX)] = llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 0), llvm::APInt(32, cfg.block.x));
    lanes[static_cast<size_t>(Lane::ThreadY)] = llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 0), llvm::APInt(32, cfg.block.y));
    lanes[static_cast<size_t>(Lane::ThreadZ)] = llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 0), llvm::APInt(32, cfg.block.z));
    lanes[static_cast<size_t>(Lane::BlockDimX)] =
        llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 1), llvm::APInt(32, cfg.block.x + 1));
    lanes[static_cast<size_t>(Lane::BlockDimY)] =
        llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 1), llvm::APInt(32, cfg.block.y + 1));
    lanes[static_cast<size_t>(Lane::BlockDimZ)] =
        llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 1), llvm::APInt(32, cfg.block.z + 1));
    // grid lanes stay unbounded (full): sound, no exact info.
  }
  return lanes;
}

// Range of an operand as seen by the lattice (constants handled directly).
std::optional<ConstantRange> lattice_range(const Lattice& lat, const Value* v,
                                           unsigned width) {
  if (auto* ci = llvm::dyn_cast<ConstantInt>(v)) {
    return ConstantRange(ci->getValue());
  }
  auto it = lat.ranges.find(v);
  if (it != lat.ranges.end()) {
    return it->second.zextOrTrunc(width);
  }
  return std::nullopt;
}

// One forward transfer step for instruction `i` (pure; may return nullopt).
std::optional<ConstantRange> transfer(const Instruction& i, const Lattice& lat) {
  const unsigned width = i.getType()->isIntegerTy()
                             ? i.getType()->getIntegerBitWidth()
                             : 64;
  auto binary = [&](auto op) -> std::optional<ConstantRange> {
    auto a = lattice_range(lat, i.getOperand(0), width);
    auto b = lattice_range(lat, i.getOperand(1), width);
    if (!a || !b) {
      return std::nullopt;
    }
    return op(*a, *b);
  };
  switch (i.getOpcode()) {
    case Instruction::Add:
      return binary([](const ConstantRange& a, const ConstantRange& b) {
        return a.add(b);
      });
    case Instruction::Sub:
      return binary([](const ConstantRange& a, const ConstantRange& b) {
        return a.sub(b);
      });
    case Instruction::Mul:
      return binary([](const ConstantRange& a, const ConstantRange& b) {
        return a.multiply(b);
      });
    case Instruction::And:
      return binary([](const ConstantRange& a, const ConstantRange& b) {
        return a.binaryAnd(b);
      });
    case Instruction::Or:
      return binary([](const ConstantRange& a, const ConstantRange& b) {
        return a.binaryOr(b);
      });
    case Instruction::Xor:
      return binary([](const ConstantRange& a, const ConstantRange& b) {
        return a.binaryXor(b);
      });
    case Instruction::Shl:
      return binary([](const ConstantRange& a, const ConstantRange& b) {
        return a.shl(b);
      });
    case Instruction::LShr:
      return binary([](const ConstantRange& a, const ConstantRange& b) {
        return a.lshr(b);
      });
    case Instruction::AShr:
      return binary([](const ConstantRange& a, const ConstantRange& b) {
        return a.ashr(b);
      });
    case Instruction::ZExt: {
      auto a = lattice_range(lat, i.getOperand(0), width);
      if (!a) {
        return std::nullopt;
      }
      return a->zextOrTrunc(width);
    }
    case Instruction::SExt: {
      auto a = lattice_range(lat, i.getOperand(0), width);
      if (!a) {
        return std::nullopt;
      }
      return a->sextOrTrunc(width);
    }
    case Instruction::Trunc: {
      auto a = lattice_range(lat, i.getOperand(0), width);
      if (!a || a->getBitWidth() <= width) {
        return std::nullopt;
      }
      return a->truncate(width);
    }
    case Instruction::ICmp:
      return ConstantRange::getFull(1);
    case Instruction::PHI: {
      const auto* phi = llvm::cast<llvm::PHINode>(&i);
      bool any = false;
      ConstantRange acc = ConstantRange::getEmpty(width);
      for (const Value* incoming : phi->incoming_values()) {
        auto r = lattice_range(lat, incoming, width);
        if (!r) {
          return std::nullopt;
        }
        any = true;
        acc = acc.unionWith(*r);
      }
      return any ? std::optional<ConstantRange>(acc) : std::nullopt;
    }
    case Instruction::Select: {
      auto t = lattice_range(lat, i.getOperand(1), width);
      auto f = lattice_range(lat, i.getOperand(2), width);
      if (!t || !f) {
        return std::nullopt;
      }
      return t->unionWith(*f);
    }
    case Instruction::GetElementPtr: {
      // Offset range: sum of the index ranges (constants folded), 64-bit.
      const auto* gep = llvm::cast<GetElementPtrInst>(&i);
      ConstantRange acc = ConstantRange::getEmpty(64);
      bool any = false;
      for (const Value* idx : gep->indices()) {
        auto r = lattice_range(lat, idx, 64);
        if (!r) {
          return std::nullopt;
        }
        any = true;
        acc = acc.unionWith(r->zextOrTrunc(64));
      }
      return any ? std::optional<ConstantRange>(acc) : std::nullopt;
    }
    default:
      return std::nullopt;
  }
}

// Runs the budgeted range propagation over the kernel. Latches the budget on
// exhaustion (the caller emits the budget-exceeded diagnostic).
void propagate(Lattice& lat, Function& f, KernelInfo& ki, DiagnosticEngine& engine) {
  std::vector<Instruction*> work;
  work.reserve(128);
  for (BasicBlock& bb : f) {
    for (Instruction& i : bb) {
      work.push_back(&i);
    }
  }
  // Seed the coordinate intrinsics from the launch config.
  for (Instruction* i : work) {
    auto* call = llvm::dyn_cast<llvm::CallInst>(i);
    if (!call) {
      continue;
    }
    auto lane = lane_for_intrinsic(call->getCalledFunction()
                                       ? call->getCalledFunction()->getName()
                                       : llvm::StringRef());
    if (lane) {
      lat.ranges.insert(
          {i, ki.intrinsic_ranges[static_cast<size_t>(*lane)]});
    }
  }
  unsigned pass = 0;
  constexpr unsigned kMaxPasses = 16;
  for (; pass < kMaxPasses; ++pass) {
    bool changed = false;
    for (Instruction* i : work) {
      if (!engine.budget().acquire(ki.name, 1)) {
        return;  // budget latched; caller emits the diagnostic
      }
      if (lat.ranges.count(i)) {
        continue;  // seeded intrinsic (fixed)
      }
      auto r = transfer(*i, lat);
      if (!r) {
        continue;
      }
      auto it = lat.ranges.find(i);
      if (it == lat.ranges.end()) {
        lat.ranges.insert({i, *r});
        changed = true;
      } else {
        ConstantRange widened = it->second.unionWith(*r);
        if (widened != it->second) {
          it->second = widened;
          changed = true;
        }
      }
    }
    if (!changed) {
      break;
    }
  }
  for (const auto& [value, r] : lat.ranges) {
    if (!r.isFullSet() && !r.isEmptySet()) {
      ++lat.bounded;
    }
  }
}

// DebugLoc-backed source location of an instruction (empty when the IR
// carries no debug info, e.g. compiled without -gline-tables-only).
SourceLoc source_loc_of(const Instruction& i) {
  SourceLoc loc;
  if (const llvm::DebugLoc dl = i.getDebugLoc()) {
    loc.file = dl->getFilename().str();
    loc.line = dl->getLine();
    loc.col = dl->getColumn();
  }
  return loc;
}

// Short human rendering of an instruction ("load i32, ptr %addr").
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

// Budgeted memory-access scan: bounds each load/store index; under -O2
// ingestion (degraded mode) unbounded indices emit a reconstruction-failed
// verdict. Returns the count of accesses with a bounded range.
size_t scan_accesses(Function& f, KernelInfo& ki, Lattice& lat,
                     DiagnosticEngine& engine, bool o2_degraded) {
  size_t bounded = 0;
  for (BasicBlock& bb : f) {
    for (Instruction& i : bb) {
      if (!engine.budget().acquire(ki.name, 1)) {
        return bounded;
      }
      Value* addr = nullptr;
      if (auto* load = llvm::dyn_cast<llvm::LoadInst>(&i)) {
        addr = load->getPointerOperand();
      } else if (auto* store = llvm::dyn_cast<llvm::StoreInst>(&i)) {
        addr = store->getPointerOperand();
      } else if (auto* rmw = llvm::dyn_cast<llvm::AtomicRMWInst>(&i)) {
        addr = rmw->getPointerOperand();
      } else if (auto* cx = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&i)) {
        addr = cx->getPointerOperand();
      } else {
        continue;
      }
      const Value* idx = addr;
      if (auto* gep = llvm::dyn_cast<GetElementPtrInst>(addr)) {
        idx = gep;
      }
      std::optional<ConstantRange> r = lattice_range(lat, idx, 64);
      if (r && r->isFullSet()) {
        r.reset();
      }
      if (r && !r->isEmptySet() && r->getUpper() != r->getLower()) {
        ++bounded;
        continue;
      }
      if (o2_degraded) {
        Diagnostic d;
        d.severity = Severity::Warning;
        d.class_id = EngineClass::kScevReconstructionFailed;
        d.function = ki.name;
        d.loc = source_loc_of(i);
        d.instruction = short_instruction(i);
        d.message = "degraded verdict: index bounds could not be reconstructed "
                    "via SCEV at -O2 for this access (verify at -O1 for exact "
                    "ranges)";
        engine.emit(std::move(d));
      }
    }
  }
  return bounded;
}

std::optional<ConstantRange> query_range_impl(const VerifierState& state,
                                              const KernelInfo& ki, Value* v,
                                              Function* owning_fn) {
  if (auto* ci = llvm::dyn_cast<ConstantInt>(v)) {
    return ConstantRange(ci->getValue());
  }
  Lattice* lat = nullptr;
  for (size_t i = 0; i < state.kernels().size(); ++i) {
    if (&state.kernels()[i] == &ki) {
      lat = const_cast<Lattice*>(&state.impl()->lattices[i]);
      break;
    }
  }
  if (lat) {
    if (auto it = lat->ranges.find(v); it != lat->ranges.end()) {
      if (!it->second.isFullSet()) {
        return it->second;
      }
    }
  }
  // SCEV fallback (research brief §4.1): loop-carried indices etc.
  if (auto* call = llvm::dyn_cast<llvm::CallInst>(v)) {
    if (auto lane = lane_for_intrinsic(call->getCalledFunction()
                                           ? call->getCalledFunction()->getName()
                                           : llvm::StringRef())) {
      const ConstantRange& r = ki.intrinsic_ranges[static_cast<size_t>(*lane)];
      if (!r.isFullSet()) {
        return r;
      }
      return std::nullopt;
    }
  }
  Function* fn = owning_fn;
  if (!fn) {
    if (auto* inst = llvm::dyn_cast<llvm::Instruction>(v)) {
      fn = inst->getFunction();
    }
  }
  if (!fn) {
    return std::nullopt;
  }
  auto& analyses = state.impl()->analyses;
  auto it = analyses.find(fn);
  if (it == analyses.end()) {
    it = analyses.emplace(fn, std::make_unique<FuncAnalysis>(*fn)).first;
  }
  ConstantRange scev = it->second->se.getUnsignedRange(it->second->se.getSCEV(v));
  if (scev.isFullSet() || scev.isEmptySet()) {
    return std::nullopt;
  }
  return scev;
}

}  // namespace

// --------------------------------------------------------------------------
// VerifierState
// --------------------------------------------------------------------------

VerifierState::VerifierState(std::shared_ptr<Module> module,
                             std::shared_ptr<llvm::LLVMContext> context,
                             VerifierOptions options)
    : impl_(new Impl),
      module_(std::move(module)),
      options_(options),
      engine_(new DiagnosticEngine(options_)) {
  (void)context;  // kept alive by the caller alongside module_
}

VerifierState::~VerifierState() = default;

void VerifierState::analyze() {
  if (analyzed_) {
    return;
  }
  for (Function* f : findKernels(*module_)) {
    KernelInfo ki;
    ki.function = f;
    ki.mangled_name = f->getName().str();
    std::string dem = llvm::demangle(ki.mangled_name);
    ki.name = (dem.empty() || dem == ki.mangled_name) ? ki.mangled_name : dem;
    ki.config = resolveLaunchConfig(*module_, *f);
    ki.intrinsic_ranges = compute_intrinsic_ranges(ki.config);
    if (ki.config.symbolic()) {
      // Tier (c) degraded verdict: "symbolic dims" + runtime-check suggestion
      // (research brief §4.6; acceptance case 2 for --device-only).
      Diagnostic d;
      d.severity = Severity::Warning;
      d.class_id = EngineClass::kSymbolicDims;
      d.function = ki.name;
      d.loc = source_loc_of(*f->begin()->begin());
      d.message = "symbolic dims: degraded verdict — launch configuration is "
                  "not a compile-time constant (or was compiled "
                  "--device-only); " +
                  ki.config.runtime_check_suggestion;
      engine_->emit(std::move(d));
    }
    kernels_.push_back(std::move(ki));
  }
  impl_->lattices.resize(kernels_.size());
  impl_->bounded_access_counts.resize(kernels_.size(), 0);
  const bool o2_degraded = options_.o2_reconstruction_degraded &&
                           options_.ingestion_opt_level >= 2;
  for (size_t k = 0; k < kernels_.size(); ++k) {
    KernelInfo& ki = kernels_[k];
    engine_->budget().reset(ki.name);
    Lattice& lat = impl_->lattices[k];
    propagate(lat, *ki.function, ki, *engine_);
    impl_->bounded_access_counts[k] =
        scan_accesses(*ki.function, ki, lat, *engine_, o2_degraded);
    impl_->total_bounded_accesses += impl_->bounded_access_counts[k];
    if (engine_->budget().exceeded(ki.name)) {
      engine_->emitBudgetExceeded(ki.name);
    }
    for (const auto& plugin : CheckRegistry::instance().plugins()) {
      VerifierContext ctx{*module_, ki, *engine_};
      ctx.state = this;
      std::vector<Diagnostic> plugin_diags = plugin->run(ctx);
      for (Diagnostic& d : plugin_diags) {
        engine_->emit(std::move(d));
      }
      if (engine_->budget().exceeded(ki.name)) {
        engine_->emitBudgetExceeded(ki.name);
      }
    }
  }
  analyzed_ = true;
}

const std::vector<KernelInfo>& VerifierState::kernels() const { return kernels_; }

const std::vector<Diagnostic>& VerifierState::diagnostics() const {
  return engine_->diagnostics();
}

VerifierState::Impl* VerifierState::impl() const { return impl_.get(); }

uint64_t VerifierState::boundedAccessCount() const {
  return impl_->total_bounded_accesses;
}

std::string VerifierState::dumpState() const {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "kernscope-verify state dump\n";
  os << "module: " << module_->getName() << " (triple "
     << module_->getTargetTriple().str() << ")\n";
  os << "kernels: " << kernels_.size() << "\n";
  for (size_t k = 0; k < kernels_.size(); ++k) {
    const KernelInfo& ki = kernels_[k];
    os << "\nkernel " << ki.name << " (" << ki.mangled_name << ")\n";
    os << "  launch tier: " << launch_tier_name(ki.config.tier) << "\n";
    os << "  origin: " << ki.config.origin << "\n";
    if (ki.config.tier == LaunchTier::Exact) {
      os << "  blockDim: " << ki.config.block.render() << " (exact)\n";
      os << "  gridDim: " << ki.config.grid.render() << " (exact)\n";
    } else if (ki.config.tier == LaunchTier::UpperBound) {
      os << "  blockDim: <= " << ki.config.block.render() << " (upper bound)\n";
      os << "  gridDim: unbounded\n";
    } else {
      os << "  blockDim: symbolic\n";
      os << "  gridDim: symbolic\n";
    }
    os << "  exact dim ranges:\n";
    bool any = false;
    for (int lane = 0; lane < 12; ++lane) {
      const ConstantRange& r = ki.intrinsic_ranges[lane];
      if (!r.isFullSet() && !r.isEmptySet()) {
        any = true;
        os << "    " << lane_name(static_cast<Lane>(lane)) << " in ";
        r.print(os);
        os << "\n";
      }
    }
    if (!any) {
      os << "    <none — symbolic dims; degraded verdict>\n";
    }
    if (options_.dump_ranges && k < impl_->lattices.size()) {
      os << "  propagated ranges: " << impl_->lattices[k].ranges.size()
         << " values, " << impl_->lattices[k].bounded << " bounded\n";
    }
    os << "  bounded memory-access indices: "
       << (k < impl_->bounded_access_counts.size() ? impl_->bounded_access_counts[k] : 0)
       << "\n";
    os << "  budget: " << engine_->budget().consumed(ki.name) << "/"
       << options_.max_states_per_function << " states\n";
  }
  os << "\nregistered checks:\n";
  for (const CheckRegistry::Entry& e : CheckRegistry::instance().enumerate()) {
    os << "  " << e.id << " (" << e.name << "): classes ";
    for (size_t i = 0; i < e.class_ids.size(); ++i) {
      if (i) {
        os << ",";
      }
      os << e.class_ids[i];
    }
    os << "\n";
  }
  if (CheckRegistry::instance().enumerate().empty()) {
    os << "  <none — check modules land in todos 31-34>\n";
  }
  os << "\ndiagnostics (" << engine_->diagnostics().size() << "):\n";
  for (const Diagnostic& d : engine_->diagnostics()) {
    os << "  " << d.render() << "\n";
  }
  os.flush();
  return text;
}

// --------------------------------------------------------------------------
// VerifierContext
// --------------------------------------------------------------------------

std::optional<ConstantRange> VerifierContext::queryRange(Value* v) {
  return query_range_impl(*state, kernel, v, kernel.function);
}

llvm::ScalarEvolution* VerifierContext::scalarEvolution() const {
  auto& analyses = state->impl()->analyses;
  auto it = analyses.find(kernel.function);
  if (it == analyses.end()) {
    it = analyses.emplace(kernel.function,
                          std::make_unique<FuncAnalysis>(*kernel.function)).first;
  }
  return &it->second->se;
}

}  // namespace verify
}  // namespace kernscope
