// kernscope::verify::oob — shared helpers implementation (todo 31).
// See oob_common.h for the soundness-scope contract.

#include "oob_common.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <algorithm>
#include <cstring>
#include <functional>

namespace kernscope {
namespace verify {
namespace oob {

using llvm::APInt;
using llvm::ConstantExpr;
using llvm::ConstantInt;
using llvm::ConstantRange;
using llvm::Function;
using llvm::GlobalVariable;
using llvm::Module;
using llvm::Type;
using llvm::Value;

namespace {

// Strips bitcast/addrspacecast instructions and constant expressions.
Value* stripCasts(Value* v) {
  while (true) {
    if (auto* ci = llvm::dyn_cast<llvm::CastInst>(v)) {
      if (ci->getOpcode() == llvm::Instruction::BitCast ||
          ci->getOpcode() == llvm::Instruction::AddrSpaceCast) {
        v = ci->getOperand(0);
        continue;
      }
    }
    if (auto* ce = llvm::dyn_cast<ConstantExpr>(v)) {
      if (ce->getOpcode() == llvm::Instruction::BitCast ||
          ce->getOpcode() == llvm::Instruction::AddrSpaceCast) {
        v = ce->getOperand(0);
        continue;
      }
    }
    return v;
  }
}

// Maps a __device_stub__ reference back to the device kernel definition
// (mirrors the core's find_kernel_for_launch: demangle-based, because naive
// mangled-name surgery is unsound).
Function* kernelForLaunch(Module& m, Function* launch_ref) {
  if (!launch_ref) {
    return nullptr;
  }
  const std::string name = launch_ref->getName().str();
  if (name.find("__device_stub__") == std::string::npos) {
    return launch_ref;  // HIP: the launch reference IS the kernel.
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

// Merges one launch-site smem size into the per-kernel map.  Any unknown
// (runtime) size poisons the kernel's bound (sound: the kernel may be
// launched with any size); multiple constant sizes merge to the minimum
// (sound: the kernel must be safe under the smallest observed launch).
void mergeSmem(SmemSizeMap& out, Function* kernel,
               std::optional<uint64_t> smem) {
  if (!kernel) {
    return;
  }
  auto it = out.find(kernel);
  if (it == out.end()) {
    out[kernel] = smem;
    return;
  }
  std::optional<uint64_t> merged;
  if (it->second && smem) {
    merged = std::min(*it->second, *smem);
  }
  out[kernel] = merged;
}

}  // namespace

std::string jsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
    }
  }
  return out;
}

uint64_t accessByteWidth(Type* ty, const llvm::Module& m) {
  return m.getDataLayout().getTypeStoreSize(ty).getFixedValue();
}

std::string range_render(const ConstantRange& r) {
  std::string s;
  llvm::raw_string_ostream os(s);
  r.print(os);
  os.flush();
  return s;
}

bool isGepChain(Value* v) {
  return llvm::isa<llvm::GetElementPtrInst>(stripCasts(v));
}

AllocationInfo resolveAllocation(Value* ptr, const llvm::Module& m) {
  AllocationInfo info;
  Value* cur = stripCasts(ptr);
  // Walk GEP pointer operands to the base value.
  while (auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(cur)) {
    cur = stripCasts(gep->getPointerOperand());
  }
  if (auto* gv = llvm::dyn_cast<GlobalVariable>(cur)) {
    info.name = gv->getName().str();
    const unsigned as = gv->getAddressSpace();
    if (as == 1) {
      info.root = AllocationInfo::Root::DeviceGlobal;
    } else if (as == 3) {
      info.root = AllocationInfo::Root::SharedArray;
    } else {
      return info;  // const (4) / other spaces: not in this module's v1 scope.
    }
    Type* ty = gv->getValueType();
    const uint64_t bytes = m.getDataLayout().getTypeAllocSize(ty).getFixedValue();
    // extern __shared__ lowers to a ZERO-SIZED global: the size comes from
    // the launch site, resolved by the caller via scanLaunchSmem.
    if (bytes == 0) {
      info.root = AllocationInfo::Root::SharedDynamic;
      info.known_size = false;
    } else {
      info.known_size = true;
      info.size_bytes = bytes;
    }
    return info;
  }
  if (llvm::isa<llvm::Argument>(cur)) {
    info.root = AllocationInfo::Root::KernelParam;
    info.name = "kernel-param";
    return info;
  }
  if (llvm::isa<llvm::AllocaInst>(cur)) {
    info.root = AllocationInfo::Root::Local;
    info.name = "alloca";
    return info;
  }
  info.name = "<unknown>";
  return info;
}

std::optional<ConstantRange> gepByteSpan(Value* ptr, VerifierContext& ctx,
                                         std::string& out_index_range,
                                         uint64_t& out_innermost_stride) {
  const llvm::DataLayout& dl = ctx.module.getDataLayout();
  out_index_range.clear();
  out_innermost_stride = 0;

  // Recursive accumulator.  The base pointer contributes the constant zero:
  // the single-point range [0, 1) is the additive identity for non-wrapping
  // ConstantRange::add ([a,b) + [0,1) == [a,b)).
  struct Acc {
    ConstantRange bytes = ConstantRange::getNonEmpty(APInt(64, 0), APInt(64, 1));
    bool valid = true;
  };
  std::function<Acc(Value*)> span_of = [&](Value* v) -> Acc {
    v = stripCasts(v);
    auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(v);
    if (!gep) {
      return Acc{};  // base pointer: zero contribution.
    }
    Acc inner = span_of(gep->getPointerOperand());
    if (!inner.valid) {
      return inner;
    }
    Type* ty = gep->getSourceElementType();
    for (Value* idx_v : gep->indices()) {
      if (auto* st = llvm::dyn_cast<llvm::StructType>(ty)) {
        auto* ci = llvm::dyn_cast<ConstantInt>(idx_v);
        if (!ci) {
          inner.valid = false;
          return inner;  // dynamic struct field index: not boundable.
        }
        const uint64_t off =
            dl.getStructLayout(st)->getElementOffset(ci->getZExtValue());
        inner.bytes = inner.bytes.add(ConstantRange(APInt(64, off)));
        ty = st->getElementType(ci->getZExtValue());
        continue;
      }
      // Array/vector/scalar level: the index is in units of the element
      // type; remember the innermost index range + stride for the proof
      // field and the last-byte width.
      Type* elt = ty;
      if (ty->isArrayTy()) {
        elt = ty->getArrayElementType();
      } else if (ty->isVectorTy()) {
        elt = llvm::cast<llvm::VectorType>(ty)->getElementType();
      }
      const uint64_t stride = dl.getTypeAllocSize(elt).getFixedValue();
      out_innermost_stride = stride;
      std::optional<ConstantRange> r = ctx.queryRange(idx_v);
      if (!r || r->isFullSet() || r->isEmptySet()) {
        inner.valid = false;
        return inner;  // unbounded index (data-dependent / runtime).
      }
      ConstantRange r64 = r->zextOrTrunc(64);
      ConstantRange scaled = r64.multiply(ConstantRange(APInt(64, stride)));
      if (scaled.isFullSet() || scaled.isEmptySet()) {
        inner.valid = false;
        return inner;  // overflowed: cannot reason about it.
      }
      inner.bytes = inner.bytes.add(scaled);
      if (inner.bytes.isFullSet() || inner.bytes.isEmptySet()) {
        inner.valid = false;
        return inner;
      }
      out_index_range = range_render(*r);
      ty = elt;
    }
    return inner;
  };

  Acc acc = span_of(ptr);
  if (!acc.valid) {
    return std::nullopt;
  }
  return acc.bytes;
}

SmemSizeMap scanLaunchSmem(Module& m) {
  SmemSizeMap out;
  for (Function& f : m) {
    if (f.isDeclaration() || f.getCallingConv() == llvm::CallingConv::PTX_Kernel ||
        f.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL) {
      continue;
    }
    // The push may live in a different basic block than the pop+launch
    // (clang: push at the end of the entry block, pop+launch under the
    // error check) — the pending smem must persist per FUNCTION.  The
    // __device_stub__ bodies ALSO contain an orphan pop+launch with the
    // runtime-loaded smem value; only the push/pop pair in the CALLER
    // carries the constant config, so orphan pops and non-constant launch
    // operands must never poison the per-kernel bound.
    std::optional<uint64_t> pending_smem;
    bool saw_push = false;
    for (llvm::BasicBlock& bb : f) {
      for (llvm::Instruction& i : bb) {
        auto* call = llvm::dyn_cast<llvm::CallInst>(&i);
        if (!call || !call->getCalledFunction()) {
          continue;
        }
        llvm::StringRef name = call->getCalledFunction()->getName();
        if (name == "__cudaPushCallConfiguration" ||
            name == "__hipPushCallConfiguration") {
          // (grid{i64 xy,i32 z}, block{i64 xy,i32 z}, smem i64, stream).
          saw_push = true;
          if (call->arg_size() >= 5) {
            if (auto* ci =
                    llvm::dyn_cast<ConstantInt>(call->getArgOperand(4))) {
              pending_smem = ci->getZExtValue();
            } else {
              pending_smem = std::nullopt;  // runtime smem: unknowable.
            }
          }
        } else if (name == "__cudaPopCallConfiguration" ||
                   name == "__hipPopCallConfiguration") {
          if (!saw_push) {
            continue;  // orphan pop inside a __device_stub__: no config.
          }
          for (llvm::Instruction& j : bb) {
            auto* launch = llvm::dyn_cast<llvm::CallInst>(&j);
            if (!launch || launch == call || !launch->getCalledFunction()) {
              continue;
            }
            llvm::StringRef lname = launch->getCalledFunction()->getName();
            if (lname == "cudaLaunchKernel" || lname == "hipLaunchKernel") {
              // The kernel is the FIRST ARGUMENT of the launch call (the
              // __device_stub__ reference), not the called function.
              const Value* func =
                  launch->getArgOperand(0)->stripPointerCasts();
              auto* kernel = llvm::dyn_cast<Function>(const_cast<Value*>(func));
              mergeSmem(out, kernelForLaunch(m, kernel), pending_smem);
              break;
            }
            if (lname.contains("__device_stub__")) {
              mergeSmem(out, kernelForLaunch(m, launch->getCalledFunction()),
                        pending_smem);
              break;
            }
          }
        } else if (name == "cudaConfigureCall") {
          // (grid, block, smem size_t, stream) + next __device_stub__ call.
          if (call->arg_size() < 3) {
            continue;
          }
          std::optional<uint64_t> smem;
          if (auto* ci = llvm::dyn_cast<ConstantInt>(call->getArgOperand(2))) {
            smem = ci->getZExtValue();
          }
          for (llvm::Instruction& j : bb) {
            auto* launch = llvm::dyn_cast<llvm::CallInst>(&j);
            if (!launch || launch == call || !launch->getCalledFunction()) {
              continue;
            }
            if (launch->getCalledFunction()->getName().contains("__device_stub__")) {
              mergeSmem(out, kernelForLaunch(m, launch->getCalledFunction()),
                        smem);
              break;
            }
          }
        } else if (name == "cudaLaunchKernel" || name == "hipLaunchKernel") {
          // Direct-style launch (no push/pop, e.g. HIP): (func, grid,
          // block, args, smem, stream).  Only a CONSTANT smem operand
          // contributes; a runtime value here is the config-clone of a
          // push/pop launch (see the function-scope comment above).
          if (call->arg_size() < 6) {
            continue;
          }
          const Value* func = call->getArgOperand(0)->stripPointerCasts();
          auto* kernel = llvm::dyn_cast<Function>(const_cast<Value*>(func));
          if (auto* ci =
                  llvm::dyn_cast<ConstantInt>(call->getArgOperand(4))) {
            mergeSmem(out, kernelForLaunch(m, kernel), ci->getZExtValue());
          }
        }
      }
    }
  }
  return out;
}

}  // namespace oob
}  // namespace verify
}  // namespace kernscope
