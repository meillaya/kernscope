//===- pad_pass.cpp - shared-memory padding pass (todo 38) ----------------===//
//
// kernscope-pad, the shared-memory bank-conflict padding pass. See pad_pass.h
// for the design; this file implements the analysis + rewrite.
//
// Stride analysis (on the smem GEPs of an addrspace(3) allocation):
//   * the lane variable is the return of llvm.nvvm.read.ptx.sreg.tid.x /
//     .laneid or llvm.amdgcn.workitem.id.x;
//   * each GEP index is classified: constant (coeff 0), affine in the lane
//     var (constant coeff, via add/sub/mul/shl and zext/sext/trunc chains),
//     or data-dependent (unprovable -> decline);
//   * a lane-affine index at array position j contributes
//       stride_bytes = coeff * sizeof(T) * prod(dims[j+1..k])
//     where dims are the nested array dimensions of the allocation;
//   * the bank footprint is computed with the analyzer's todo-16 formula
//     (tools/analyze/src/analyze_banks.cpp compute_banks): over one warp of
//     32 lanes, bank = (word/4) mod 32, way = max multiplicity. way > 1 is a
//     provable conflict.
//
// The +1 pad grows the innermost dimension (dims[k] -> dims[k]+1), which
// rescues every NON-innermost stride (the padded row is one element longer,
// breaking the repeated-bank pattern). The rewrite is a type change
// propagated through the whole GEP chain: the allocation type and every GEP
// source type on the allocation's array-type spine are replaced consistently
// (indices are untouched — the same index vector now addresses the padded
// layout). In-place for allocas (setAllocatedType); globals get a new
// GlobalVariable + RAUW (LLVM 21 GlobalVariable has no type setter). A shared
// global's GEPs are collected MODULE-WIDE: a shared array is one allocation,
// so it is padded once and every accessing kernel's GEPs are rewritten
// together — never a partial rewrite.
//
// libs/arch reuse: the smem budget guard calls kernscope::get_arch /
// has_arch (kernscope/arch.h, header-only INTERFACE lib) — no limit is
// hardcoded here. The arch is resolved from the "kernscope.pad.arch" function
// attribute (deterministic test hook), then from the "nvvmarch" /
// "amdgpu-arch" module flags clang emits for real CUDA/HIP compiles; if
// neither is present the pass declines (never silently assumes a budget).
// Bank-footprint constants (32 banks, 4-byte words, 32 lanes) are the same
// values libs/arch encodes (warp_size 32 on every supported row; the brief
// §2.1 model), kept in one place here because libs/arch does not itself
// contain a bank-footprint function — the analyzer's compute_banks is the
// canonical implementation of the shared formula and is cited as such.
//
//===----------------------------------------------------------------------===//

#include "pad_pass.h"

#include <kernscope/arch.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include <kernscope/pass/register_pass.h>

namespace kernscope {
namespace pass {
namespace {

// ---------------------------------------------------------------------------
// Bank model (shared with the analyzer's todo-16 module — see pad_pass.h).
// ---------------------------------------------------------------------------

constexpr int kNumBanks = 32;  // banks per SM / CU (brief §2.1, §2.2)
constexpr int kBytesPerBank = 4;

// Bank footprint of one warp of `lanes` lanes over addresses
// base_word + stride_words*lane (words). Way = max multiplicity per bank;
// way 1 = conflict-free (including stride 0 broadcast). Identical loop to
// tools/analyze/src/analyze_banks.cpp compute_banks (analyzer todo 16).
struct BankFootprint {
  int64_t stride_words = 0;
  int64_t stride_mod_32 = 0;
  int distinct_banks = 0;
  int way = 0;
  bool resolved = false;  // false: stride not 4-byte word aligned
};

BankFootprint computeBankFootprint(int64_t stride_bytes, int lanes) {
  BankFootprint r;
  if (stride_bytes % kBytesPerBank != 0) return r;  // not word-aligned
  r.stride_words = stride_bytes / kBytesPerBank;
  r.stride_mod_32 = ((r.stride_words % kNumBanks) + kNumBanks) % kNumBanks;

  if (r.stride_words == 0) {  // broadcast: all lanes, same word
    r.distinct_banks = 1;
    r.way = 1;
    r.resolved = true;
    return r;
  }

  std::map<int, int> bank_count;
  for (int lane = 0; lane < lanes; ++lane) {
    const int64_t word = r.stride_words * lane;
    const int bank =
        static_cast<int>((word % kNumBanks + kNumBanks) % kNumBanks);
    ++bank_count[bank];
  }
  r.distinct_banks = static_cast<int>(bank_count.size());
  r.way = 0;
  for (const auto &[bank, count] : bank_count)
    r.way = std::max(r.way, count);
  r.resolved = true;
  return r;
}

// ---------------------------------------------------------------------------
// Arch resolution + smem budget (libs/arch reuse — never hardcoded).
// ---------------------------------------------------------------------------

// Arch for the smem budget: the "kernscope.pad.arch" function attribute
// (deterministic test hook), else the nvvmarch / amdgpu-arch module flags
// clang emits for real CUDA/HIP compiles, else nullopt (decline).
std::optional<kernscope::ArchInfo> resolveArch(const llvm::Module &M,
                                               const llvm::Function *F) {
  if (F) {
    const llvm::Attribute A = F->getFnAttribute("kernscope.pad.arch");
    if (A.isStringAttribute() && kernscope::has_arch(A.getValueAsString()))
      return kernscope::get_arch(A.getValueAsString());
  }
  for (const char *Key : {"nvvmarch", "amdgpu-arch"}) {
    const llvm::Metadata *Flag = M.getModuleFlag(Key);
    if (Flag && llvm::isa<llvm::MDString>(Flag)) {
      const std::string Name = llvm::cast<llvm::MDString>(Flag)
                                   ->getString()
                                   .str();
      if (kernscope::has_arch(Name)) return kernscope::get_arch(Name);
    }
  }
  return std::nullopt;
}

// Per-block smem budget of the arch: the opt-in per-block cap when modeled,
// else the per-SM/per-CU shared memory (AMD CDNA rows model no opt-in cap).
int64_t smemBudget(const kernscope::ArchInfo &A) {
  return A.smem_per_block_max_bytes > 0 ? A.smem_per_block_max_bytes
                                        : A.smem_per_sm_bytes;
}

// ---------------------------------------------------------------------------
// Lane-variable + affine-index classification.
// ---------------------------------------------------------------------------

bool isLaneIntrinsic(const llvm::Function *Callee) {
  if (!Callee || !Callee->isIntrinsic()) return false;
  const llvm::StringRef Name = Callee->getName();
  return Name == "llvm.nvvm.read.ptx.sreg.tid.x" ||
         Name == "llvm.nvvm.read.ptx.sreg.laneid" ||
         Name == "llvm.amdgcn.workitem.id.x";
}

// The lane variable of F: the result of the first lane intrinsic. (Mixing
// tid.x and laneid in one function is pathological; the fixtures use exactly
// one, and a function with none yields nullptr — every index then classifies
// as a constant.)
llvm::Value *laneVarOf(const llvm::Function &F) {
  for (const llvm::BasicBlock &BB : F)
    for (const llvm::Instruction &I : BB)
      if (const auto *Call = llvm::dyn_cast<llvm::CallInst>(&I))
        if (isLaneIntrinsic(Call->getCalledFunction()))
          return const_cast<llvm::Instruction *>(&I);
  return nullptr;
}

// Classify V as c0 + c1*laneVar. Returns the coefficient c1 when V is
// statically affine in exactly one lane variable; nullopt when V is
// data-dependent (not provable). Constant values classify as coeff 0.
std::optional<int64_t> classifyAffine(llvm::Value *V, llvm::Value *LaneVar) {
  if (V == LaneVar) return 1;
  if (llvm::isa<llvm::ConstantInt>(V)) return 0;

  if (auto *I = llvm::dyn_cast<llvm::Instruction>(V)) {
    switch (I->getOpcode()) {
      case llvm::Instruction::Add: {
        const auto A = classifyAffine(I->getOperand(0), LaneVar);
        const auto B = classifyAffine(I->getOperand(1), LaneVar);
        if (!A || !B) return std::nullopt;
        return *A + *B;
      }
      case llvm::Instruction::Sub: {
        const auto A = classifyAffine(I->getOperand(0), LaneVar);
        const auto B = classifyAffine(I->getOperand(1), LaneVar);
        if (!A || !B) return std::nullopt;
        return *A - *B;
      }
      case llvm::Instruction::Mul: {
        const auto A = classifyAffine(I->getOperand(0), LaneVar);
        const auto B = classifyAffine(I->getOperand(1), LaneVar);
        if (!A || !B) return std::nullopt;
        if (*A != 0 && *B != 0 &&
            *A > std::numeric_limits<int64_t>::max() / *B)
          return std::nullopt;  // overflow: not statically known
        return *A * *B;
      }
      case llvm::Instruction::Shl: {
        const auto A = classifyAffine(I->getOperand(0), LaneVar);
        auto *Shift = llvm::dyn_cast<llvm::ConstantInt>(I->getOperand(1));
        if (!A || !Shift) return std::nullopt;
        if (Shift->getZExtValue() >= 63) return std::nullopt;
        if (*A > std::numeric_limits<int64_t>::max() >>
                     static_cast<unsigned>(Shift->getZExtValue()))
          return std::nullopt;
        return *A << static_cast<unsigned>(Shift->getZExtValue());
      }
      case llvm::Instruction::ZExt:
      case llvm::Instruction::SExt:
      case llvm::Instruction::Trunc:
      case llvm::Instruction::BitCast:
        return classifyAffine(I->getOperand(0), LaneVar);
      default:
        return std::nullopt;
    }
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Allocation candidates.
// ---------------------------------------------------------------------------

// One addrspace(3) array allocation (alloca or internal global) with its
// nested dimensions and scalar element type.
struct Allocation {
  llvm::Value *Value = nullptr;    // AllocaInst or GlobalVariable
  llvm::Type *AllocType = nullptr; // ArrayType (possibly nested)
  llvm::Type *ElemType = nullptr;  // scalar element
  llvm::SmallVector<uint64_t, 4> Dims;
  bool IsGlobal = false;

  static std::optional<Allocation> make(llvm::Value *V) {
    llvm::Type *T = nullptr;
    bool isGlobal = false;
    if (auto *A = llvm::dyn_cast<llvm::AllocaInst>(V)) {
      if (A->getAddressSpace() != 3) return std::nullopt;
      T = A->getAllocatedType();
    } else if (auto *G = llvm::dyn_cast<llvm::GlobalVariable>(V)) {
      if (G->getAddressSpace() != 3 || !G->hasInternalLinkage())
        return std::nullopt;
      isGlobal = true;
      T = G->getValueType();
    } else {
      return std::nullopt;
    }
    if (!T) return std::nullopt;

    Allocation A;
    A.Value = V;
    A.IsGlobal = isGlobal;
    A.AllocType = T;
    // Unwrap the nested array type: [d0 x [d1 x ... [dk x T]]].
    while (auto *AT = llvm::dyn_cast<llvm::ArrayType>(T)) {
      A.Dims.push_back(AT->getNumElements());
      T = AT->getElementType();
    }
    A.ElemType = T;
    if (A.Dims.empty()) return std::nullopt;  // scalar smem: nothing to pad
    return A;
  }

  // Total elements of the +1-padded layout (innermost dim +1).
  int64_t paddedElements() const {
    int64_t N = 1;
    for (size_t i = 0; i + 1 < Dims.size(); ++i)
      N *= static_cast<int64_t>(Dims[i]);
    N *= static_cast<int64_t>(Dims.back()) + 1;
    return N;
  }
};

// Build the padded array type: innermost dim +1, outer dims preserved.
llvm::Type *padArrayType(llvm::Type *T) {
  llvm::SmallVector<uint64_t, 8> Dims;
  llvm::Type *Cur = T;
  while (auto *AT = llvm::dyn_cast<llvm::ArrayType>(Cur)) {
    Dims.push_back(AT->getNumElements());
    Cur = AT->getElementType();
  }
  if (Dims.empty()) return T;  // scalar: no array to pad
  ++Dims.back();               // the +1 column pad
  llvm::Type *Res = Cur;
  for (auto It = Dims.rbegin(); It != Dims.rend(); ++It)
    Res = llvm::ArrayType::get(Res, *It);
  return Res;
}
llvm::Type *paddedAllocType(const Allocation &A) {
  return padArrayType(A.AllocType);
}

// The array-type spine of the allocation: every nested array type along the
// nesting (plus the scalar element type), mapped to its padded version. GEP
// source types are rewritten through this map; the scalar maps to itself.
void buildTypeMap(llvm::Type *AllocType,
                  std::map<llvm::Type *, llvm::Type *> &Map) {
  llvm::Type *T = AllocType;
  while (auto *AT = llvm::dyn_cast<llvm::ArrayType>(T)) {
    Map[T] = padArrayType(T);
    T = AT->getElementType();
  }
  Map[T] = T;  // scalar element type: identity
}

// ---------------------------------------------------------------------------
// GEP-chain collection (module-wide): every GEP on the allocation reachable
// through addrspacecast/bitcast constant expressions (the pattern clang
// emits for __shared__ arrays). A shared global's chain spans every kernel
// that accesses it, so the pad is applied once and rewritten consistently.
// ---------------------------------------------------------------------------

// Walk the users of `Root`; collect all reachable GEPs into `Geps` and set
// `HasUnsupported` if a chain node (the allocation itself or a cast) has a
// user that is neither a GEP nor a passthrough cast (call argument,
// ptrtoint, pointer store...). Users of a GEP result are ordinary
// loads/stores and need no rewriting (opaque pointers). The collection is
// not truncated by obstacles: the flag is what makes the caller decline
// before any mutation.
void collectGeps(llvm::Value *Root,
                 llvm::SmallVectorImpl<llvm::GetElementPtrInst *> &Geps,
                 bool &HasUnsupported) {
  llvm::SmallVector<llvm::Value *, 16> Worklist;
  Worklist.push_back(Root);
  while (!Worklist.empty()) {
    llvm::Value *V = Worklist.pop_back_val();
    // GEP results are opaque pointers: any consumer (load/store/...) is
    // fine. Only the allocation and cast chain nodes need strict users.
    const bool IsChainNode = !llvm::isa<llvm::GetElementPtrInst>(V);
    for (llvm::User *U : V->users()) {
      if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(U)) {
        Geps.push_back(GEP);
        Worklist.push_back(GEP);
        continue;
      }
      if (llvm::isa<llvm::AddrSpaceCastInst>(U) ||
          llvm::isa<llvm::BitCastInst>(U)) {
        Worklist.push_back(U);
        continue;
      }
      // Constant-expression casts (e.g. the clang addrspacecast of the
      // shared global): follow their users; as opaque-ptr constants they
      // need no rewrite themselves (RAUW retargets them).
      if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(U)) {
        if (CE->getOpcode() == llvm::Instruction::AddrSpaceCast ||
            CE->getOpcode() == llvm::Instruction::BitCast) {
          Worklist.push_back(CE);
          continue;
        }
      }
      if (IsChainNode) HasUnsupported = true;
    }
  }
}

// ---------------------------------------------------------------------------
// Per-candidate analysis + rewrite.
// ---------------------------------------------------------------------------

enum class DeclineReason {
  kNoConflict,      // no provable conflict (or no lane-affine stride at all)
  kDataDependent,   // some GEP index is not statically affine
  kUnaligned,       // stride not 4-byte aligned (unmodelable)
  kInnermostOnly,   // conflict only from the innermost stride (+1 pad can't fix)
  kNoArch,          // no arch attribute/flag -> cannot verify the budget
  kWave64,          // arch is wave64 LDS (CDNA) — wave32 model only
  kBudget,          // padded size would exceed the arch smem budget
  kUnsupportedUser, // a non-GEP user of the allocation exists
};

const char *declineReasonText(DeclineReason R) {
  switch (R) {
    case DeclineReason::kNoConflict:
      return "no provable bank conflict";
    case DeclineReason::kDataDependent:
      return "data-dependent index; conflict not provable";
    case DeclineReason::kUnaligned:
      return "stride not 4-byte aligned; bank footprint unmodelable";
    case DeclineReason::kInnermostOnly:
      return "conflict from innermost stride not fixable by +1 pad";
    case DeclineReason::kNoArch:
      return "no arch (kernscope.pad.arch attribute or nvvmarch/amdgpu-arch "
             "module flag); smem budget unverifiable";
    case DeclineReason::kWave64:
      return "arch is wave64 LDS (AMD CDNA); wave32 model only";
    case DeclineReason::kBudget:
      return "padded size exceeds smem budget";
    case DeclineReason::kUnsupportedUser:
      return "unsupported user (not a GEP) of the allocation";
  }
  return "unknown";
}

struct AnalysisResult {
  bool Changed = false;
  bool Declined = false;
  DeclineReason Reason = DeclineReason::kNoConflict;
  int WorstWay = 0;
  int64_t PaddedBytes = 0;
  int64_t BudgetBytes = 0;
  std::string ArchName;
};

// Decide whether the allocation can be padded, and if so rewrite it. `Geps`
// is the module-wide GEP chain collected from the allocation; `F` is the
// function the lane-variable analysis anchors on (the allocation's owner for
// allocas, any accessing kernel for shared globals); `HasUnsupported` records
// whether a non-GEP user exists (call argument, ptrtoint, pointer store...),
// which makes the rewrite unsafe. All analysis happens BEFORE the first
// mutation (two-phase: a late decline leaves IR untouched).
AnalysisResult processAllocation(llvm::Module &M, const llvm::Function *F,
                                 Allocation &Alloc,
                                 const llvm::SmallVectorImpl<
                                     llvm::GetElementPtrInst *> &Geps,
                                 bool HasUnsupported,
                                 const llvm::DataLayout &DL) {
  AnalysisResult R;
  if (Geps.empty()) {  // no smem GEPs -> nothing to fix
    R.Declined = true;
    R.Reason = DeclineReason::kNoConflict;
    return R;
  }
  if (HasUnsupported) {
    R.Declined = true;
    R.Reason = DeclineReason::kUnsupportedUser;
    return R;
  }

  const int64_t ElemSize = DL.getTypeAllocSize(Alloc.ElemType);

  // Spine position of a GEP source type: 0 for the full allocation type,
  // k for the innermost array, k+1 for the scalar element type.
  auto spinePosition = [&](llvm::Type *T) -> std::optional<size_t> {
    if (T == Alloc.AllocType) return 0;
    if (T == Alloc.ElemType) return Alloc.Dims.size();
    size_t P = 1;
    llvm::Type *Cur = Alloc.AllocType;
    while (auto *AT = llvm::dyn_cast<llvm::ArrayType>(Cur)) {
      if (T == AT->getElementType()) return P;
      Cur = AT->getElementType();
      ++P;
    }
    return std::nullopt;
  };

  int WorstWay = 1;
  bool HasFixableConflict = false;
  for (llvm::GetElementPtrInst *GEP : Geps) {
    const auto Pos = spinePosition(GEP->getSourceElementType());
    if (!Pos) {  // GEP source type not on the allocation's spine
      R.Declined = true;
      R.Reason = DeclineReason::kUnsupportedUser;
      return R;
    }
    llvm::Value *LaneVar = laneVarOf(*GEP->getFunction());
    // Stride per index: exactly what the GEP computes — each index scales
    // by the alloc size of the type it indexes into (e.g. the row index of
    // `gep [16 x [32 x float]], ..., 0, %row` scales by 128 bytes). The
    // innermost index is the one indexing the scalar element type; the +1
    // pad changes every other stride (fixable) but never that one.
    for (auto IT = llvm::gep_type_begin(GEP),
              ET = llvm::gep_type_end(GEP);
         IT != ET; ++IT) {
      const auto Coeff = classifyAffine(IT.getOperand(), LaneVar);
      if (!Coeff) {
        R.Declined = true;
        R.Reason = DeclineReason::kDataDependent;
        return R;
      }
      if (*Coeff == 0) continue;
      const bool IsInnermost = (IT.getIndexedType() == Alloc.ElemType);
      const int64_t Stride = *Coeff * DL.getTypeAllocSize(IT.getIndexedType());
      const BankFootprint B =
          computeBankFootprint(Stride, kNumBanks);  // 32 lanes (wave32)
      if (!B.resolved) {
        R.Declined = true;
        R.Reason = DeclineReason::kUnaligned;
        return R;
      }
      WorstWay = std::max(WorstWay, B.way);
      if (B.way > 1 && !IsInnermost) HasFixableConflict = true;
    }
  }
  if (!HasFixableConflict) {
    R.Declined = true;
    R.Reason = WorstWay > 1 ? DeclineReason::kInnermostOnly
                            : DeclineReason::kNoConflict;
    R.WorstWay = WorstWay;
    return R;
  }

  // Applicability gate 4: the padded layout must fit the arch smem budget.
  const auto Arch = resolveArch(M, F);
  if (!Arch) {
    R.Declined = true;
    R.Reason = DeclineReason::kNoArch;
    R.WorstWay = WorstWay;
    return R;
  }
  R.ArchName = std::string(Arch->name);
  R.BudgetBytes = smemBudget(*Arch);
  if (Arch->warp_size != kNumBanks) {  // wave64 LDS (AMD CDNA) not modeled
    R.Declined = true;
    R.Reason = DeclineReason::kWave64;
    R.WorstWay = WorstWay;
    return R;
  }

  const int64_t PaddedBytes = Alloc.paddedElements() * ElemSize;
  R.PaddedBytes = PaddedBytes;
  if (PaddedBytes > R.BudgetBytes) {
    R.Declined = true;
    R.Reason = DeclineReason::kBudget;
    R.WorstWay = WorstWay;
    return R;
  }

  // ---- Rewrite: allocation type + every GEP source type on the spine. ----
  llvm::Type *NewAllocType = paddedAllocType(Alloc);
  std::map<llvm::Type *, llvm::Type *> TypeMap;
  buildTypeMap(Alloc.AllocType, TypeMap);

  if (auto *A = llvm::dyn_cast<llvm::AllocaInst>(Alloc.Value)) {
    A->setAllocatedType(NewAllocType);
  } else if (auto *G = llvm::dyn_cast<llvm::GlobalVariable>(Alloc.Value)) {
    llvm::Constant *Init =
        G->hasInitializer() ? G->getInitializer() : nullptr;
    if (Init && !llvm::isa<llvm::ConstantAggregateZero>(Init)) {
      // A non-zero-initialized shared global cannot be re-shaped safely.
      R.Declined = true;
      R.Reason = DeclineReason::kUnsupportedUser;
      return R;
    }
    const std::string OrigName = G->getName().str();
    auto *NewG = new llvm::GlobalVariable(
        M, NewAllocType, G->isConstant(), G->getLinkage(),
        Init ? llvm::ConstantAggregateZero::get(NewAllocType) : nullptr,
        OrigName, /*InsertBefore=*/nullptr,
        llvm::GlobalVariable::NotThreadLocal, G->getAddressSpace(),
        G->isExternallyInitialized());
    G->replaceAllUsesWith(NewG);
    G->eraseFromParent();
    NewG->setName(OrigName);  // drop the creation-time uniquify suffix
    Alloc.Value = NewG;
  }

  // Rebuild every GEP with the padded source type, preserving nowrap flags.
  // (Every source type is on the spine after the analysis above, so the map
  // lookup cannot fail here — the two-phase order guarantees it.)
  for (llvm::GetElementPtrInst *GEP : Geps) {
    auto It = TypeMap.find(GEP->getSourceElementType());
    if (It == TypeMap.end()) {
      R.Declined = true;
      R.Reason = DeclineReason::kUnsupportedUser;
      return R;
    }
    llvm::SmallVector<llvm::Value *, 4> Idx(GEP->indices().begin(),
                                            GEP->indices().end());
    auto *NewGEP = llvm::GetElementPtrInst::Create(
        It->second, GEP->getPointerOperand(), Idx, GEP->getNoWrapFlags(),
        GEP->getName(), GEP->getIterator());
    GEP->replaceAllUsesWith(NewGEP);
    GEP->eraseFromParent();
  }

  R.Changed = true;
  R.WorstWay = WorstWay;
  return R;
}

}  // namespace

llvm::PreservedAnalyses SharedMemoryPadPass::run(llvm::Module &M,
                                                 llvm::ModuleAnalysisManager &) {
  const llvm::DataLayout &DL = M.getDataLayout();
  bool Changed = false;;

  // Candidates: addrspace(3) allocas (per function), and internal
  // addrspace(3) globals (once per global — a shared array is one allocation
  // even when several kernels access it).
  for (llvm::Function &F : M) {
    if (F.isDeclaration()) continue;
    for (llvm::BasicBlock &BB : F)
      for (llvm::Instruction &I : BB) {
        auto *A = llvm::dyn_cast<llvm::AllocaInst>(&I);
        if (!A || A->getAddressSpace() != 3) continue;
        auto AllocOpt = Allocation::make(A);
        if (!AllocOpt) continue;
        Allocation Alloc = *AllocOpt;

        llvm::SmallVector<llvm::GetElementPtrInst *, 8> Geps;
        bool HasUnsupported = false;
        collectGeps(Alloc.Value, Geps, HasUnsupported);

        AnalysisResult R =
            processAllocation(M, &F, Alloc, Geps, HasUnsupported, DL);
        if (R.Changed) {
          Changed = true;
        } else if (R.Declined && R.Reason != DeclineReason::kNoConflict) {
          llvm::errs() << "kernscope-pad: declining pad of " << A->getName()
                       << " in @" << F.getName() << ": "
                       << declineReasonText(R.Reason);
          if (R.Reason == DeclineReason::kBudget)
            llvm::errs() << " (" << R.PaddedBytes << " bytes, arch budget "
                         << R.BudgetBytes << " for " << R.ArchName << ")";
          llvm::errs() << "\n";
        }
      }
  }

  // Candidate globals are collected first: the rewrite replaces the
  // GlobalVariable, so the iteration must not touch the module global list.
  llvm::SmallVector<llvm::GlobalVariable *, 8> SharedGlobals;
  for (llvm::GlobalVariable &G : M.globals())
    if (G.getAddressSpace() == 3 && G.hasInternalLinkage())
      SharedGlobals.push_back(&G);
  for (llvm::GlobalVariable *G : SharedGlobals) {
    auto AllocOpt = Allocation::make(G);
    if (!AllocOpt) continue;
    Allocation Alloc = *AllocOpt;

    llvm::SmallVector<llvm::GetElementPtrInst *, 8> Geps;
    bool HasUnsupported = false;
    collectGeps(Alloc.Value, Geps, HasUnsupported);

    // Anchor the lane analysis on the first accessing kernel (arch comes
    // from its attribute; module flags are the fallback).
    const llvm::Function *F = nullptr;
    for (llvm::GetElementPtrInst *GEP : Geps)
      if (GEP->getFunction()) {
        F = GEP->getFunction();
        break;
      }
    if (!F) continue;  // dead global (no in-function users)

    AnalysisResult R = processAllocation(M, F, Alloc, Geps, HasUnsupported,
                                         DL);
    if (R.Changed) {
      Changed = true;
    } else if (R.Declined && R.Reason != DeclineReason::kNoConflict) {
      llvm::errs() << "kernscope-pad: declining pad of " << G->getName()
                   << ": " << declineReasonText(R.Reason);
      if (R.Reason == DeclineReason::kBudget)
        llvm::errs() << " (" << R.PaddedBytes << " bytes, arch budget "
                     << R.BudgetBytes << " for " << R.ArchName << ")";
      llvm::errs() << "\n";
    }
  }

  return Changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

}  // namespace pass
}  // namespace kernscope

namespace {

bool parsePadPass(llvm::StringRef Name, llvm::ModulePassManager &MPM,
                  llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
  if (Name == "kernscope-pad") {
    MPM.addPass(kernscope::pass::SharedMemoryPadPass());
    return true;
  }
  return false;
}

}  // namespace

// Self-registration (todo-36 contract — tools/pass/README.md). The pad pass
// is module-scoped, so the textual name and the pipeline-start EP both add
// it straight to the ModulePassManager. The peephole EP is NOT registered:
// allocation rewrites are module-scope transforms, not peepholes, and
// re-running them after each InstCombine is neither useful nor safe.
REGISTER_PASS_CALLBACK([](llvm::PassBuilder &PB) {
  PB.registerPipelineParsingCallback(parsePadPass);

  PB.registerPipelineStartEPCallback(
      [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) {
        MPM.addPass(kernscope::pass::SharedMemoryPadPass());
      });
});
