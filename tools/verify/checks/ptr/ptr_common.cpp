// kernscope::verify::ptr — shared helpers for the pointer-discipline check
// module (todo 34). See ptr_common.h for the soundness scope.

#include "ptr_common.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/raw_ostream.h>

#include <unordered_set>

namespace kernscope {
namespace verify {
namespace ptr {

namespace {

using llvm::GetElementPtrInst;
using llvm::GEPOperator;

// Strips pointer-to-pointer casts (bitcast/addrspacecast/ptrtoint chains),
// including CONSTANT EXPRESSION casts (clang 21 folds casts into
// ConstantExpr operands, e.g. `addrspacecast (ptr addrspace(1) @g_buf to
// ptr)` inside a GEP's pointer operand or an inline GEP expression).
const llvm::Value* stripCasts(const llvm::Value* v) {
  while (true) {
    if (const auto* bc = llvm::dyn_cast<llvm::BitCastInst>(v)) {
      v = bc->getOperand(0);
    } else if (const auto* ac = llvm::dyn_cast<llvm::AddrSpaceCastInst>(v)) {
      v = ac->getOperand(0);
    } else if (const auto* pt = llvm::dyn_cast<llvm::PtrToIntInst>(v)) {
      v = pt->getOperand(0);
    } else if (const auto* ce = llvm::dyn_cast<const llvm::ConstantExpr>(v)) {
      switch (ce->getOpcode()) {
        case llvm::Instruction::BitCast:
        case llvm::Instruction::AddrSpaceCast:
        case llvm::Instruction::PtrToInt:
          v = ce->getOperand(0);
          break;
        default:
          return v;
      }
    } else {
      return v;
    }
  }
}

}  // namespace

AllocationInfo resolveAllocation(const llvm::Value* ptr, const llvm::Module& m) {
  AllocationInfo info;
  const llvm::Value* v = stripCasts(ptr);
  if (const auto* gep = llvm::dyn_cast<GEPOperator>(v)) {
    v = stripCasts(gep->getPointerOperand());
  }
  const llvm::DataLayout& dl = m.getDataLayout();
  if (const auto* gv = llvm::dyn_cast<const llvm::GlobalVariable>(v)) {
    info.name = gv->getName().str();
    const unsigned as = gv->getAddressSpace();
    if (as == 1) {
      info.root = AllocationInfo::Root::DeviceGlobal;
    } else if (as == 3) {
      info.root = AllocationInfo::Root::SharedArray;
    } else if (as == 4) {
      // __constant__: treated as a known-size device-side allocation for the
      // pointer-discipline checks (like DeviceGlobal).
      info.root = AllocationInfo::Root::DeviceGlobal;
    } else {
      info.root = AllocationInfo::Root::Unknown;
      return info;
    }
    if (gv->getValueType()->isSized()) {
      info.size_bytes = dl.getTypeAllocSize(gv->getValueType());
      if (info.size_bytes > 0) {
        info.known_size = true;
      }
    }
    return info;
  }
  if (const auto* arg = llvm::dyn_cast<const llvm::Argument>(v)) {
    if (arg->getType()->isPointerTy()) {
      info.root = AllocationInfo::Root::KernelParam;
      info.name = arg->getName().str();
    }
    return info;
  }
  if (const auto* alloca = llvm::dyn_cast<const llvm::AllocaInst>(v)) {
    info.root = AllocationInfo::Root::Local;
    info.name = alloca->getName().str();
    if (alloca->getAllocatedType()->isSized()) {
      info.size_bytes = dl.getTypeAllocSize(alloca->getAllocatedType());
      info.known_size = true;
    }
    return info;
  }
  return info;
}

bool isGepChain(llvm::Value* v) {
  const llvm::Value* base = stripCasts(v);
  return llvm::isa<GetElementPtrInst>(base);
}

std::optional<llvm::ConstantRange> gepByteSpan(llvm::Value* ptr,
                                               VerifierContext& ctx,
                                               std::string& out_index_range,
                                               uint64_t& out_innermost_stride) {
  out_index_range.clear();
  out_innermost_stride = 0;
  const llvm::Value* base = stripCasts(ptr);
  const auto* gep = llvm::dyn_cast<GEPOperator>(base);
  if (gep == nullptr) {
    return std::nullopt;
  }
  const llvm::Module& m = ctx.module;
  const llvm::DataLayout& dl = m.getDataLayout();

  // Collect the GEP chain innermost-first (GEPOperator: both instructions
  // and ConstantExpr GEPs, which clang 21 folds into operands).
  llvm::SmallVector<const GEPOperator*, 4> chain;
  for (const GEPOperator* g = gep; g != nullptr;) {
    chain.push_back(g);
    const llvm::Value* p = stripCasts(g->getPointerOperand());
    g = llvm::dyn_cast<GEPOperator>(p);
  }

  // Byte span [lo, hi): iterate levels outermost-first, accumulating
  // (index-range * stride). All offsets are unsigned (negative indices wrap
  // into a huge span -> conservatively flagged by the caller).
  llvm::APInt lo(64, 0), hi(64, 1);
  bool first = true;
  const auto* innermost_gep = chain.front();
  llvm::SmallVector<unsigned, 4> idxs;
  for (unsigned k = 0; k < innermost_gep->getNumIndices(); ++k) {
    idxs.push_back(k);
  }
  // Determine strides per level using gep_type_iterator semantics: level k
  // strides the type indexed at that level.
  for (size_t ci = chain.size(); ci-- > 0;) {
    const GEPOperator* g = chain[ci];
    llvm::SmallVector<unsigned, 4> var_idxs;
    llvm::SmallVector<uint64_t, 4> strides;
    llvm::Type* cur = g->getSourceElementType();
    uint64_t running_stride = 1;
    // Walk the indices; for each, compute its stride in bytes.
    // Struct indices contribute a constant offset (handled by folding into
    // lo/hi via the constant operand); array indices stride element size.
    for (unsigned idx = 0; idx < g->getNumIndices(); ++idx) {
      const llvm::Value* idxv = g->getOperand(idx + 1);
      // Get the type this index strides over (LLVM GEP semantics).
      llvm::Type* indexed_ty = nullptr;
      if (auto* st = llvm::dyn_cast<llvm::StructType>(cur)) {
        // Struct index must be a constant; fold its field offset.
        const auto* c = llvm::dyn_cast<llvm::ConstantInt>(idxv);
        if (c == nullptr) {
          return std::nullopt;  // Non-constant struct index: unanalyzable.
        }
        const unsigned field = c->getZExtValue();
        const llvm::StructLayout* sl = dl.getStructLayout(st);
        const uint64_t off = sl->getElementOffset(field);
        // Folding a constant offset: add to lo/hi span (as exact [off, off+1)
        // contribution — but the byte span accumulates via range add; exact
        // offsets widen lo/hi by the field width only at the leaf level.
        // Simplify: constant struct offsets fold into the span bounds.
        // Represent via the index range [off, off+1) * stride 1 at this
        // level by adjusting lo/hi directly:
        if (first) {
          lo = llvm::APInt(64, off);
          hi = llvm::APInt(64, off + 1);
          first = false;
        } else {
          lo += off;
          hi += off;
        }
        cur = st->getElementType(field);
        continue;
      }
      if (auto* at = llvm::dyn_cast<llvm::ArrayType>(cur)) {
        indexed_ty = at->getElementType();
        running_stride = dl.getTypeAllocSize(indexed_ty);
      } else if (auto* vt = llvm::dyn_cast<llvm::VectorType>(cur)) {
        indexed_ty = vt->getElementType();
        running_stride = dl.getTypeAllocSize(indexed_ty);
      } else {
        indexed_ty = cur;
        running_stride = dl.getTypeAllocSize(cur);
      }
      if (const auto* c = llvm::dyn_cast<llvm::ConstantInt>(idxv)) {
        // Constant index: fold its offset exactly.
        const llvm::APInt& cv = c->getValue();
        if (first) {
          lo = cv.zextOrTrunc(64) * running_stride;
          hi = lo + running_stride;
          first = false;
        } else {
          lo += cv.zextOrTrunc(64) * running_stride;
          hi += cv.zextOrTrunc(64) * running_stride;
        }
      } else {
        // Variable index: query its range; widen the span by
        // [range.lo, range.hi) * stride.
        std::optional<llvm::ConstantRange> r =
            ctx.queryRange(const_cast<llvm::Value*>(idxv));
        if (!r) {
          return std::nullopt;  // Unbounded index: caller does may-analysis.
        }
        const llvm::APInt rlo = r->getLower().zextOrTrunc(64);
        const llvm::APInt rhi = r->getUpper().zextOrTrunc(64);
        if (first) {
          lo = rlo * running_stride;
          hi = rhi * running_stride;
          first = false;
        } else {
          lo += rlo * running_stride;
          hi += rhi * running_stride;
        }
        if (ci == 0) {
          // Innermost variable index range, rendered for the proof field.
          out_index_range = range_render(*r);
          out_innermost_stride = running_stride;
        }
      }
      cur = indexed_ty;
    }
  }
  if (first) {
    lo = llvm::APInt(64, 0);
    hi = llvm::APInt(64, 1);
  }
  return llvm::ConstantRange::getNonEmpty(lo, hi);
}

uint64_t accessByteWidth(llvm::Type* ty, const llvm::Module& m) {
  return m.getDataLayout().getTypeAllocSize(ty);
}

SmemSizeMap scanLaunchSmem(llvm::Module& m) {
  SmemSizeMap out;
  // Scan host functions for the smem argument of the launch forms:
  //   cudaConfigureCall(grid, block, smem, stream)
  //   __cudaPushCallConfiguration(grid, block, smem, stream, ...)
  //   cudaLaunchKernel(f, grid, block, smem, stream, ...)
  for (llvm::Function& f : m) {
    for (llvm::BasicBlock& bb : f) {
      for (llvm::Instruction& i : bb) {
        auto* ci = llvm::dyn_cast<llvm::CallInst>(&i);
        if (ci == nullptr) {
          continue;
        }
        const llvm::Function* callee = ci->getCalledFunction();
        if (callee == nullptr) {
          continue;
        }
        const llvm::StringRef n = callee->getName();
        int smem_arg = -1;
        llvm::Value* kernel_arg = nullptr;
        if (n == "cudaConfigureCall" || n == "__cudaPushCallConfiguration") {
          smem_arg = 2;
          if (n == "cudaConfigureCall") {
            kernel_arg = nullptr;  // kernel launched separately
          } else {
            kernel_arg = nullptr;
          }
        } else if (n == "cudaLaunchKernel") {
          smem_arg = 3;
          kernel_arg = ci->getArgOperand(0);
        } else {
          continue;
        }
        if (smem_arg < 0 ||
            static_cast<size_t>(smem_arg) >= ci->arg_size()) {
          continue;
        }
        llvm::Value* smem = ci->getArgOperand(smem_arg);
        std::optional<uint64_t> sz;
        if (const auto* c = llvm::dyn_cast<llvm::ConstantInt>(smem)) {
          sz = c->getZExtValue();
        }
        // Bind to the kernel (for cudaLaunchKernel) or mark "unknown kernel"
        // (configure forms need the kernel from a following cudaLaunchKernel
        // — resolve via a second scan below).
        llvm::Function* target = nullptr;
        if (kernel_arg != nullptr) {
          if (const auto* kc = llvm::dyn_cast<llvm::ConstantExpr>(kernel_arg)) {
            target = llvm::dyn_cast<llvm::Function>(kc->getOperand(0));
          } else if (const auto* bc =
                         llvm::dyn_cast<llvm::BitCastInst>(kernel_arg)) {
            target = llvm::dyn_cast<llvm::Function>(
                const_cast<llvm::Value*>(stripCasts(bc->getOperand(0))));
          }
        }
        if (target != nullptr) {
          out[target] = sz;
        }
        // Track pending configure-form launches for the second pass.
        (void)n;
      }
    }
  }
  return out;
}

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

std::string range_render(const llvm::ConstantRange& r) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "[" << r.getLower() << "," << r.getUpper() << ")";
  os.flush();
  return s;
}

}  // namespace ptr
}  // namespace verify
}  // namespace kernscope
