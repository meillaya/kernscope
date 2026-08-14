#include <kernscope/verify/launch_config.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <kernscope/verify/verifier_state.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <optional>
#include <string>

namespace kernscope {
namespace verify {

using llvm::CallInst;
using llvm::Constant;
using llvm::ConstantAggregateZero;
using llvm::ConstantExpr;
using llvm::ConstantInt;
using llvm::ConstantStruct;
using llvm::Function;
using llvm::Instruction;
using llvm::Module;
using llvm::StringRef;

std::string Dim3::render() const {
  return std::to_string(x) + "x" + std::to_string(y) + "x" + std::to_string(z);
}

const char* launch_tier_name(LaunchTier t) {
  switch (t) {
    case LaunchTier::Exact:
      return "Exact";
    case LaunchTier::UpperBound:
      return "UpperBound";
    case LaunchTier::Symbolic:
      return "Symbolic";
  }
  return "Unknown";
}

std::string LaunchConfig::render() const {
  std::string out = std::string(launch_tier_name(tier));
  out += " block ";
  if (tier == LaunchTier::UpperBound) {
    out += "<= ";
  }
  out += block.render();
  out += " grid ";
  if (tier == LaunchTier::Exact) {
    out += grid.render();
  } else if (tier == LaunchTier::UpperBound) {
    out += "unbounded";
  } else {
    out += "symbolic";
  }
  out += " @ ";
  out += origin.empty() ? "?" : origin;
  if (!runtime_check_suggestion.empty()) {
    out += " | " + runtime_check_suggestion;
  }
  return out;
}

namespace {

// Constant-folds a dim3 value (packed i64 lo32=x/hi32=y + i32 z; constant
// struct; aggregate zero). Returns std::nullopt when not a compile-time
// constant (-> tier Symbolic).
std::optional<Dim3> try_extract_dim3(const llvm::Value* v) {
  using namespace llvm;
  if (auto* ci = dyn_cast<ConstantInt>(v)) {
    const uint64_t val = ci->getZExtValue();
    if (ci->getBitWidth() == 64) {
      return Dim3{val & 0xffffffffULL, (val >> 32) & 0xffffffffULL, 1};
    }
    if (ci->getBitWidth() == 32) {
      return Dim3{val, 1, 1};
    }
    return std::nullopt;
  }
  if (auto* cs = dyn_cast<ConstantStruct>(v)) {
    Dim3 d;
    const unsigned n = std::min<unsigned>(3, cs->getNumOperands());
    for (unsigned i = 0; i < n; ++i) {
      auto* f = dyn_cast<ConstantInt>(cs->getOperand(i));
      if (!f) {
        return std::nullopt;
      }
      (&d.x)[i] = f->getZExtValue();
    }
    return d;
  }
  if (isa<ConstantAggregateZero>(v)) {
    return Dim3{0, 0, 0};
  }
  return std::nullopt;
}

// A dim3 whose x is 0 is not a real launch config (CUDA rejects it);
// treat as non-constant so we never claim exact dims from junk.
bool sane_dims(const Dim3& d) { return d.x != 0 && d.y != 0 && d.z != 0; }

// Source location of a call instruction (DebugLoc; may be invalid when the
// host pass carried no debug info).
SourceLoc loc_of(const llvm::Instruction* i) {
  SourceLoc loc;
  if (const llvm::DebugLoc dl = i->getDebugLoc()) {
    loc.file = dl->getFilename().str();
    loc.line = dl->getLine();
    loc.col = dl->getColumn();
  }
  return loc;
}

// One host-side launch record: kernel + dims + site.
struct LaunchRecord {
  Function* kernel = nullptr;   // null when the launch site is orphaned
  std::optional<Dim3> grid;
  std::optional<Dim3> block;
  SourceLoc loc;
};

// cudaLaunchKernel/hipLaunchKernel argument conventions:
//   (func, gridDim{i64,i32}, blockDim{i64,i32}, args, smem, stream)
bool scan_launch_kernel(CallInst* call, LaunchRecord& record) {
  if (call->arg_size() < 4) {
    return false;
  }
  const llvm::Value* func = call->getArgOperand(0)->stripPointerCasts();
  if (auto* fn = llvm::dyn_cast<llvm::Function>(const_cast<llvm::Value*>(func))) {
    record.kernel = fn;
  }
  record.grid = try_extract_dim3(call->getArgOperand(1));
  record.block = try_extract_dim3(call->getArgOperand(2));
  if (record.block && !sane_dims(*record.block)) {
    record.block.reset();
  }
  if (record.grid && !sane_dims(*record.grid)) {
    record.grid.reset();
  }
  return true;
}

// Attaches the kernel to a push-config record: clang lowers a launch site to
//   __cudaPushCallConfiguration(const dims)          (entry block)
//   __cudaPopCallConfiguration(...)  cudaLaunchKernel(&__device_stub__K, ...)
// (same basic block, pop first). The kernel is the launch call that follows
// the pop in the same block.
void attach_launch_kernel(CallInst* launch, LaunchRecord& record) {
  const llvm::Value* raw = launch->getArgOperand(0);
  const llvm::Value* func = raw->stripPointerCasts();
  std::string ty;
  llvm::raw_string_ostream os(ty);
  raw->printAsOperand(os);
  os << " " << *raw->getType();
  os.flush();
  if (auto* fn = llvm::dyn_cast<llvm::Function>(const_cast<llvm::Value*>(func))) {
      record.kernel = fn;
  }
}

// Legacy cudaConfigureCall(grid, block, smem, stream): the kernel call is
// the next stub call in the same block.
void scan_configure_call(CallInst* call, std::vector<LaunchRecord>& records) {
  if (call->arg_size() < 2) {
    return;
  }
  LaunchRecord record;
  record.grid = try_extract_dim3(call->getArgOperand(0));
  record.block = try_extract_dim3(call->getArgOperand(1));
  if (record.block && !sane_dims(*record.block)) {
    record.block.reset();
  }
  if (record.grid && !sane_dims(*record.grid)) {
    record.grid.reset();
  }
  record.loc = loc_of(call);
  for (llvm::Instruction& i : *call->getParent()) {
    auto* kcall = llvm::dyn_cast<CallInst>(&i);
    if (!kcall || kcall == call) {
      continue;
    }
    StringRef name = kcall->getCalledFunction() ? kcall->getCalledFunction()->getName()
                                                : StringRef();
    if (name.contains("__device_stub__")) {
      record.kernel = kcall->getCalledFunction();
      break;
    }
  }
  records.push_back(std::move(record));
}

// Scans the whole module for launch records (host-side only; kernels are
// excluded because they never carry launch sites). Push/pop config calls are
// paired in program order (the pop consumes the most recent unconsumed
// push, mirroring the runtime's thread-local stack), attaching the kernel
// that follows the pop.
std::vector<LaunchRecord> collect_launch_records(Module& m) {
  std::vector<LaunchRecord> records;
  for (Function& f : m) {
    if (f.isDeclaration() || f.getCallingConv() == llvm::CallingConv::PTX_Kernel ||
        f.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
      continue;
    }
    std::vector<LaunchRecord> pending;
    for (llvm::BasicBlock& bb : f) {
      for (Instruction& i : bb) {
        auto* call = llvm::dyn_cast<CallInst>(&i);
        if (!call || !call->getCalledFunction()) {
          continue;
        }
        StringRef name = call->getCalledFunction()->getName();
        if (name == "cudaLaunchKernel" || name == "hipLaunchKernel") {
          LaunchRecord record;
          if (scan_launch_kernel(call, record)) {
            record.loc = loc_of(call);
            records.push_back(std::move(record));
          }
        } else if (name == "__cudaPushCallConfiguration" ||
                   name == "__hipPushCallConfiguration") {
          if (call->arg_size() < 4) {
            continue;
          }
          // Arg layout: (grid{i64 xy,i32 z}, block{i64 xy,i32 z}, smem, stream).
          auto extract_xyz = [&](unsigned packed_idx,
                                 unsigned z_idx) -> std::optional<Dim3> {
            auto dims = try_extract_dim3(call->getArgOperand(packed_idx));
            if (!dims) {
              return std::nullopt;
            }
            if (auto* z = llvm::dyn_cast<ConstantInt>(call->getArgOperand(z_idx))) {
              dims->z = z->getZExtValue();
            }
            return dims;
          };
          LaunchRecord record;
          record.grid = extract_xyz(0, 1);
          record.block = extract_xyz(2, 3);
          if (record.block && !sane_dims(*record.block)) {
            record.block.reset();
          }
          if (record.grid && !sane_dims(*record.grid)) {
            record.grid.reset();
          }
          record.loc = loc_of(call);
          pending.push_back(std::move(record));
        } else if (name == "__cudaPopCallConfiguration" ||
                   name == "__hipPopCallConfiguration") {
          if (pending.empty()) {
            continue;
          }
          LaunchRecord record = std::move(pending.back());
          pending.pop_back();
          for (Instruction& j : bb) {
            auto* launch = llvm::dyn_cast<CallInst>(&j);
            if (!launch || launch == call || !launch->getCalledFunction()) {
              continue;
            }
            StringRef lname = launch->getCalledFunction()->getName();
            if (lname == "cudaLaunchKernel" || lname == "hipLaunchKernel") {
              attach_launch_kernel(launch, record);
              break;
            }
            if (lname.contains("__device_stub__")) {
              record.kernel = launch->getCalledFunction();
              break;
            }
          }
          records.push_back(std::move(record));
        } else if (name == "cudaConfigureCall") {
          scan_configure_call(call, records);
        }
      }
    }
  }
  return records;
}

// Maps a launch-kernel stub reference back to the device kernel definition.
// clang emits `_Z24__device_stub__lb_kernelPii` for kernel `_Z9lb_kernelPii`
// (HIP references the kernel directly, no stub). Matching is demangle-based
// (naive mangled-name surgery is unsound: parameter letters like 'P' are
// alphanumeric and cannot be told apart from identifier characters).
Function* find_kernel_for_launch(Module& m, Function* launch_ref) {
  if (!launch_ref) {
    return nullptr;
  }
  const std::string name = launch_ref->getName().str();
  if (name.find("__device_stub__") == std::string::npos) {
    return launch_ref;  // HIP: the launch ref IS the kernel.
  }
  const std::string dem = llvm::demangle(name);
  constexpr const char* kMarker = "__device_stub__";
  const size_t marker = dem.find(kMarker);
  if (marker == std::string::npos) {
    return launch_ref;
  }
  const std::string want = dem.substr(marker + std::strlen(kMarker));
  for (Function& f : m) {
    if (f.isDeclaration()) {
      continue;
    }
    if (llvm::demangle(f.getName().str()) == want) {
      return &f;
    }
  }
  return launch_ref;
}

// __launch_bounds__ / .maxntid: the nvvm.maxntid function attribute, "256"
// or "256,8" (x[,y]); z defaults to 1.
std::optional<Dim3> maxntid_from(Function& f) {
  llvm::Attribute attr = f.getFnAttribute("nvvm.maxntid");
  if (!attr.isValid()) {
    return std::nullopt;
  }
  StringRef value = attr.getValueAsString();
  Dim3 d{1, 1, 1};
  llvm::SmallVector<StringRef, 3> parts;
  value.split(parts, ',');
  if (parts.empty() || parts[0].empty()) {
    return std::nullopt;
  }
  unsigned parsed[3] = {0, 0, 0};
  for (size_t i = 0; i < parts.size() && i < 3; ++i) {
    if (parts[i].getAsInteger(10, parsed[i])) {
      return std::nullopt;
    }
  }
  d.x = parsed[0];
  if (parts.size() > 1) {
    d.y = parsed[1];
  }
  if (parts.size() > 2) {
    d.z = parsed[2];
  }
  if (d.x == 0) {
    return std::nullopt;
  }
  return d;
}

}  // namespace

std::vector<Function*> VerifierState::findKernels(Module& m) {
  std::vector<Function*> out;
  for (Function& f : m) {
    if (f.isDeclaration()) {
      continue;
    }
    const llvm::CallingConv::ID cc = f.getCallingConv();
    if (cc == llvm::CallingConv::PTX_Kernel || cc == llvm::CallingConv::AMDGPU_KERNEL ||
        f.hasFnAttribute("kernel")) {
      out.push_back(&f);
    }
  }
  return out;
}

LaunchConfig VerifierState::resolveLaunchConfig(Module& m, Function& kernel) {
  LaunchConfig config;
  config.tier = LaunchTier::Symbolic;

  // Tier (a): constant dims at a launch site naming this kernel.
  for (const LaunchRecord& rec : collect_launch_records(m)) {
    Function* target = find_kernel_for_launch(m, rec.kernel);
    if (target != &kernel) {
      continue;
    }
    if (rec.grid && rec.block) {
      config.tier = LaunchTier::Exact;
      config.grid = *rec.grid;
      config.block = *rec.block;
      config.origin = "launch site " + rec.loc.render();
      return config;
    }
  }

  // Tier (b): nvvm.maxntid / __launch_bounds__ — sound upper bound.
  if (auto maxntid = maxntid_from(kernel)) {
    config.tier = LaunchTier::UpperBound;
    config.block = *maxntid;
    config.grid = Dim3{0, 0, 0};  // unbounded
    config.origin = "nvvm.maxntid (__launch_bounds__)";
    return config;
  }

  // Tier (c): symbolic.
  config.tier = LaunchTier::Symbolic;
  config.block = Dim3{0, 0, 0};
  config.grid = Dim3{0, 0, 0};
  config.origin = "symbolic (no constant launch site)";
  config.runtime_check_suggestion =
      "launch dims are runtime values: add a host-side runtime check (e.g. "
      "assert(blockDim.x <= N && gridDim.x <= M) before the launch) to upgrade "
      "this verdict to an exact tier";
  return config;
}

}  // namespace verify
}  // namespace kernscope
