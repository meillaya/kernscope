//===-- amd_parser.cpp - AMDGPU ISA text parser (LLVM MC) ----------------===//
//
// Implementation of kernscope::amd: parse AMDGPU assembly *text* for
// gfx9/gfx11 (GCN/CDNA/RDNA) into typed instruction records + CFG, using
// LLVM 21's MC layer.
//
// Pipeline (mirrors llvm-mc's AssembleInput, llvm/tools/llvm-mc/llvm-mc.cpp):
//   1. Register all targets / asm parsers (InitializeAll*).
//   2. TargetRegistry::lookupTarget("amdgcn-...") + createMCRegInfo /
//      createMCAsmInfo / createMCSubtargetInfo / createMCInstrInfo /
//      createMCObjectFileInfo.
//   3. A custom MCStreamer records emitLabel() (labels -> CFG) and
//      emitInstruction() (MCInst -> typed AmdInstruction). The AMDGPU null
//      target streamer is attached so target-level directives
//      (.amdhsa_kernel, .amdgpu_metadata, ...) parse cleanly.
//   4. createMCAsmParser(SourceMgr, MCContext, Streamer, MAI) + the AMDGPU
//      MCTargetAsmParser; a SourceMgr diagnostic handler converts any MC
//      error into AmdParseError{message, line, triple}.
//
// Scope fallback (documented decision, per todo-10): this drives LLVM's own
// AMDGPU parser, so anything hipcc/clang -S emits for gfx9/gfx11 parses by
// construction. Hand-written fixtures use real RDNA3/CDNA ISA encodings. If
// an exotic hand-written snippet ever trips an MC quirk, the accepted scope
// is "hipcc/clang -S output only" -- that is exactly the analyzer's input
// contract. No SASS parsing, no ROCm device-library linking.
//
// Known MC quirk handled here (brittleness note): hipcc/clang 22 emits
// `.amdgcn_target "amdgcn-amd-amdhsa--gfx1101"`. LLVM 21.1's AMDGPUAsmParser
// validates that directive against the TargetID it computes from the STI;
// when the triple embeds the CPU ("amdgcn-amd-amdhsa--gfx1101") the computed
// TargetID differs ("...-unknown-gfx1101-gfx1101") and llvm-mc rejects the
// file. Verified by experiment: passing the *bare* triple "amdgcn-amd-amdhsa"
// and the CPU to createMCSubtargetInfo() separately makes the TargetID match
// exactly and everything parses. That is what this parser does.
//
//===----------------------------------------------------------------------===//

#include "kernscope/amd.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace llvm;
using namespace kernscope;

namespace {

//------------------------------------------------------------------------------
// Diagnostic capture: the SourceMgr diagnostic handler converts any MC
// error/warning into a structured record so we can throw AmdParseError with
// line + triple (todo-10 QA contract).
//------------------------------------------------------------------------------
struct DiagCapture {
  bool saw_error = false;
  std::string message;
  unsigned line = 0;
  unsigned column = 0;
};

void diag_handler(const SMDiagnostic &Diag, void *Ctx) {
  auto *Cap = static_cast<DiagCapture *>(Ctx);
  if (Diag.getKind() == SourceMgr::DK_Error && !Cap->saw_error) {
    Cap->saw_error = true;
    Cap->message = Diag.getMessage().str();
    Cap->line = static_cast<unsigned>(std::max(Diag.getLineNo(), 0));
    Cap->column = static_cast<unsigned>(std::max(Diag.getColumnNo(), 0));
  }
}

//------------------------------------------------------------------------------
// Custom MCStreamer: records labels and instructions while the MC parser
// runs. All other directive handling falls through to MCStreamer defaults
// (harmless no-ops / bookkeeping), and the AMDGPU null target streamer is
// attached for the target-level directives.
//------------------------------------------------------------------------------
struct LabelEvent {
  std::string name;
  unsigned line = 0;
};

struct Event {
  enum Kind : std::uint8_t { Label, Instr } kind = Instr;
  std::string label;      // for Label events
  std::size_t instr = 0;  // for Instr events (index into the module)
};

class AmdRecordStreamer final : public MCStreamer {
public:
  AmdRecordStreamer(MCContext &Ctx, const MCRegisterInfo &MRI,
                    const SourceMgr &SrcMgr, MCInstPrinter *Printer)
      : MCStreamer(Ctx), MRI(MRI), SrcMgr(SrcMgr), Printer(Printer) {}

  void attachParser(MCAsmParser *P) { Parser = P; }

  // ---- recording interface ----------------------------------------------
  const std::vector<AmdInstruction> &instructions() const { return Instrs; }
  const std::vector<Event> &events() const { return Events; }
  const std::vector<LabelEvent> &labels() const { return Labels; }

  // ---- MCStreamer overrides ---------------------------------------------
  void emitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override {
    Events.push_back(Event{Event::Label, Symbol->getName().str(), 0});
    Labels.push_back(LabelEvent{Symbol->getName().str(), line_for(Loc)});
    // Base bookkeeping (sets fragment/section) so the .amdhsa_kernel
    // prologue check and label expressions work.
    MCStreamer::emitLabel(Symbol, Loc);
  }

  void emitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override {
    (void)STI;
    Events.push_back(Event{Event::Instr, {}, Instrs.size()});
    Instrs.push_back(record_instruction(Inst));
    // Base default is a no-op for our purposes; don't call it (it would
    // attempt instruction encoding via MCAssembler, which we don't use).
  }

  bool emitSymbolAttribute(MCSymbol *Symbol,
                           MCSymbolAttr Attribute) override {
    (void)Symbol;
    (void)Attribute;
    // Return true = success: the generic parser treats a false return as
    // "unable to emit symbol attribute" and aborts the parse.
    return true;
  }

  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override {
    (void)Symbol;
    (void)Size;
    (void)ByteAlignment;
  }

private:
  /// 1-based source line for an SMLoc; 0 if unresolvable.
  unsigned line_for(SMLoc Loc) const {
    if (!Loc.isValid()) {
      return 0;
    }
    return SrcMgr.FindLineNumber(Loc);
  }

  /// Best-effort line of the instruction being parsed right now (the lexer is
  /// positioned at the current token, which is on the instruction's line).
  unsigned current_line() const {
    if (Parser != nullptr) {
      return SrcMgr.FindLineNumber(Parser->getLexer().getLoc());
    }
    return 0;
  }

  /// Convert one MCInst into a typed AmdInstruction.
  AmdInstruction record_instruction(const MCInst &Inst) const {
    AmdInstruction Rec;
    // The lowercase assembly mnemonic (classifies by prefix); MCInstrInfo's
    // name would be the tablegen opcode ("V_MOV_B32_e32_gfx11"). Tablegen
    // appends a trailing space after the mnemonic -> trim it.
    Rec.mnemonic = std::string(StringRef(Printer->getMnemonic(Inst).first)
                                   .rtrim(" \t"));
    Rec.klass = classify(Rec.mnemonic);
    Rec.line = current_line();
    Rec.is_tensor = Rec.klass == InstrClass::MFMA ||
                    Rec.klass == InstrClass::WMMA;
    Rec.is_scratch = Rec.klass == InstrClass::Scratch;

    // load/store / memory kind flags from the mnemonic.
    if (Rec.klass == InstrClass::LDS) {
      Rec.is_load = Rec.mnemonic.starts_with("ds_read");
      Rec.is_store = Rec.mnemonic.starts_with("ds_write");
    } else {
      const bool HasLoad = Rec.mnemonic.find("_load_") != std::string::npos;
      const bool HasStore = Rec.mnemonic.find("_store_") != std::string::npos;
      Rec.is_load = HasLoad;
      Rec.is_store = HasStore;
    }

    // control-flow metadata from the mnemonic.
    Rec.is_unconditional = Rec.mnemonic == "s_branch";
    Rec.is_conditional = Rec.mnemonic.starts_with("s_cbranch");
    Rec.is_call = Rec.mnemonic == "s_call_b64" ||
                  Rec.mnemonic == "s_swappc_b64" ||
                  Rec.mnemonic == "s_setpc_b64";
    Rec.is_return = Rec.mnemonic == "s_endpgm" || Rec.mnemonic == "s_return";
    Rec.is_branch = Rec.is_unconditional || Rec.is_conditional || Rec.is_call;

    // operands (typed) + full canonical text.
    std::ostringstream Text;
    Text << Rec.mnemonic;
    for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
      const MCOperand &Op = Inst.getOperand(I);
      AmdOperand AOp;
      if (Op.isReg()) {
        AOp.kind = AmdOperand::Kind::Register;
        AOp.text = std::string(MRI.getName(Op.getReg()));
      } else if (Op.isImm()) {
        AOp.kind = AmdOperand::Kind::Immediate;
        AOp.text = std::to_string(Op.getImm());
      } else if (Op.isSFPImm()) {
        AOp.kind = AmdOperand::Kind::Immediate;
        SmallString<16> Buf;
        raw_svector_ostream OS(Buf);
        OS << format("%g", static_cast<double>(Op.getSFPImm()));
        AOp.text = Buf.str().str();
      } else if (Op.isDFPImm()) {
        AOp.kind = AmdOperand::Kind::Immediate;
        SmallString<16> Buf;
        raw_svector_ostream OS(Buf);
        OS << format("%g", Op.getDFPImm());
        AOp.text = Buf.str().str();
      } else if (Op.isExpr()) {
        const MCExpr *E = Op.getExpr();
        if (const auto *SRE = dyn_cast<MCSymbolRefExpr>(E)) {
          AOp.kind = AmdOperand::Kind::SymbolRef;
          AOp.text = SRE->getSymbol().getName().str();
          // Branch/call target = the symbol operand.
          if (Rec.is_branch && !Rec.branch_target) {
            Rec.branch_target = SRE->getSymbol().getName().str();
          }
        } else {
          AOp.kind = AmdOperand::Kind::Expression;
          SmallString<32> Buf;
          raw_svector_ostream OS(Buf);
          getContext().getAsmInfo()->printExpr(OS, *E);
          AOp.text = Buf.str().str();
        }
      } else {
        AOp.kind = AmdOperand::Kind::Expression;
        AOp.text = "<other>";
      }
      Rec.operands.push_back(std::move(AOp));
      if (I != 0) {
        Text << ", ";
      }
      Text << Rec.operands.back().text;
    }
    Rec.text = Text.str();
    return Rec;
  }

  static InstrClass classify(std::string_view M) {
    if (M.starts_with("v_mfma")) {
      return InstrClass::MFMA; // CDNA2/3 (brief §5)
    }
    if (M.starts_with("v_wmma")) {
      return InstrClass::WMMA; // RDNA3 (brief §5)
    }
    if (M.starts_with("v_")) {
      return InstrClass::VALU;
    }
    if (M.starts_with("s_branch") || M.starts_with("s_cbranch") ||
        M == "s_call_b64" || M == "s_swappc_b64" || M == "s_setpc_b64") {
      return InstrClass::Branch;
    }
    if (M == "s_barrier") {
      return InstrClass::Barrier;
    }
    if (M.starts_with("s_waitcnt")) {
      return InstrClass::Wait;
    }
    if (M == "s_endpgm" || M == "s_return") {
      return InstrClass::EndPgm;
    }
    if (M.starts_with("s_")) {
      return InstrClass::SALU;
    }
    if (M.starts_with("ds_")) {
      return InstrClass::LDS;
    }
    if (M.starts_with("global_")) {
      return InstrClass::Global;
    }
    if (M.starts_with("buffer_")) {
      return InstrClass::Buffer;
    }
    if (M.starts_with("flat_")) {
      return InstrClass::Flat;
    }
    if (M.starts_with("scratch_")) {
      return InstrClass::Scratch; // SPILL class (brief §6)
    }
    return InstrClass::Unknown;
  }

  const MCRegisterInfo &MRI;
  const SourceMgr &SrcMgr;
  MCAsmParser *Parser = nullptr;
  MCInstPrinter *Printer;

  std::vector<AmdInstruction> Instrs;
  std::vector<Event> Events;
  std::vector<LabelEvent> Labels;
};

//------------------------------------------------------------------------------
// CFG construction: labels + terminator instructions partition the linear
// instruction stream into basic blocks; branch-target symbols and implicit
// fall-through give the successor edges.
//------------------------------------------------------------------------------
std::vector<AmdBlock> build_cfg(const std::vector<AmdInstruction> &Instrs,
                                const std::vector<Event> &Events) {
  std::vector<AmdBlock> Blocks;
  std::size_t Generated = 0;
  auto fresh_name = [&]() {
    return "<bb" + std::to_string(Generated++) + ">";
  };
  auto new_block = [&](const std::string &Name) {
    AmdBlock B;
    B.name = Name.empty() ? fresh_name() : Name;
    B.is_entry = Blocks.empty();
    Blocks.push_back(std::move(B));
    return Blocks.size() - 1;
  };

  std::size_t Cur = new_block("");
  bool closed = false; // true = the next instruction starts a fresh block

  auto open_if_needed = [&]() {
    if (!closed) {
      return;
    }
    Cur = new_block("");
    closed = false;
  };

  for (const Event &E : Events) {
    if (E.kind == Event::Label) {
      // A label always starts a fresh block (named by the label).
      Cur = new_block(E.label);
      closed = false;
      continue;
    }
    open_if_needed();
    Blocks[Cur].instruction_index.push_back(E.instr);
    const AmdInstruction &Ins = Instrs[E.instr];
    // A terminator (return / unconditional / conditional branch) closes the
    // current block.
    closed = Ins.is_return || Ins.is_unconditional || Ins.is_conditional;
  }
  if (Blocks.empty()) {
    return Blocks;
  }

  // Name map for successor resolution.
  std::map<std::string, std::size_t> NameToIndex;
  for (std::size_t I = 0; I < Blocks.size(); ++I) {
    NameToIndex[Blocks[I].name] = I;
  }

  // Successor edges.
  for (std::size_t I = 0; I < Blocks.size(); ++I) {
    AmdBlock &B = Blocks[I];
    const bool HasNext = I + 1 < Blocks.size();
    auto add_succ = [&](std::string_view Name) {
      auto It = NameToIndex.find(std::string(Name));
      if (It != NameToIndex.end() && It->second != I) {
        B.successors.push_back(Blocks[It->second].name);
      }
    };

    if (B.instruction_index.empty()) {
      // Empty block (e.g. a lone label): still falls through.
      B.has_fallthrough = HasNext;
      continue;
    }
    const AmdInstruction &Last = Instrs[B.instruction_index.back()];
    if (Last.is_return) {
      B.has_fallthrough = false; // s_endpgm terminates the kernel
    } else if (Last.is_unconditional) {
      B.has_fallthrough = false;
      if (Last.branch_target) {
        add_succ(*Last.branch_target);
      }
    } else if (Last.is_conditional || Last.is_call) {
      // Conditional branch and call both keep the target + fall-through edge.
      B.has_fallthrough = true;
      if (Last.branch_target) {
        add_succ(*Last.branch_target);
      }
      if (HasNext) {
        add_succ(Blocks[I + 1].name);
      }
    } else {
      B.has_fallthrough = HasNext;
      if (HasNext) {
        add_succ(Blocks[I + 1].name);
      }
    }
  }

  // Dedup successors, then reverse edges for predecessors.
  for (AmdBlock &B : Blocks) {
    std::sort(B.successors.begin(), B.successors.end());
    B.successors.erase(std::unique(B.successors.begin(), B.successors.end()),
                       B.successors.end());
  }
  for (std::size_t I = 0; I < Blocks.size(); ++I) {
    for (const std::string &S : Blocks[I].successors) {
      auto It = NameToIndex.find(S);
      if (It != NameToIndex.end()) {
        Blocks[It->second].predecessors.push_back(Blocks[I].name);
      }
    }
  }
  return Blocks;
}

//------------------------------------------------------------------------------
// LLVM MC component ownership (RAII): all created objects must outlive the
// parser Run() call; the components are freed in reverse order afterwards.
//------------------------------------------------------------------------------
struct McSession {
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<MCSubtargetInfo> STI;
  std::unique_ptr<MCInstrInfo> MII;
  std::unique_ptr<MCObjectFileInfo> MOFI;
  std::unique_ptr<MCInstPrinter> IP;
  std::unique_ptr<MCStreamer> Streamer;
  std::unique_ptr<MCAsmParser> Parser;
  std::unique_ptr<MCTargetAsmParser> TAP;
};

/// One-time registration of all LLVM targets + asm parsers. This is
/// required for TargetRegistry::lookupTarget("amdgcn-...") to succeed.
void ensure_targets_initialized() {
  static std::once_flag Once;
  std::call_once(Once, [] {
    InitializeAllTargetInfos();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
  });
}

} // namespace

namespace kernscope {

//------------------------------------------------------------------------------
// AmdParseError
//------------------------------------------------------------------------------
AmdParseError::AmdParseError(const std::string &what,
                             const std::string &triple,
                             const std::string &msg, unsigned ln,
                             unsigned col)
    : std::runtime_error(what), target_triple(triple), detail(msg), line(ln),
      column(col) {}

//------------------------------------------------------------------------------
// AmdModule helpers
//------------------------------------------------------------------------------
std::size_t AmdModule::count_class(InstrClass c) const {
  std::size_t N = 0;
  for (const AmdInstruction &I : instructions) {
    if (I.klass == c) {
      ++N;
    }
  }
  return N;
}

std::vector<std::string> AmdModule::mnemonics(InstrClass c) const {
  std::vector<std::string> Out;
  for (const AmdInstruction &I : instructions) {
    if (I.klass == c) {
      Out.push_back(I.mnemonic);
    }
  }
  std::sort(Out.begin(), Out.end());
  Out.erase(std::unique(Out.begin(), Out.end()), Out.end());
  return Out;
}

bool AmdModule::has_tensor_instruction() const {
  return tensor_instruction_count() > 0;
}

std::size_t AmdModule::tensor_instruction_count() const {
  std::size_t N = 0;
  for (const AmdInstruction &I : instructions) {
    if (I.is_tensor) {
      ++N;
    }
  }
  return N;
}

const char *instr_class_name(InstrClass c) {
  switch (c) {
  case InstrClass::Unknown: return "Unknown";
  case InstrClass::VALU:    return "VALU";
  case InstrClass::SALU:    return "SALU";
  case InstrClass::LDS:     return "LDS";
  case InstrClass::Global:  return "Global";
  case InstrClass::Buffer:  return "Buffer";
  case InstrClass::Flat:    return "Flat";
  case InstrClass::Scratch: return "Scratch";
  case InstrClass::Branch:  return "Branch";
  case InstrClass::Barrier: return "Barrier";
  case InstrClass::Wait:    return "Wait";
  case InstrClass::EndPgm:  return "EndPgm";
  case InstrClass::MFMA:    return "MFMA";
  case InstrClass::WMMA:    return "WMMA";
  }
  return "Unknown";
}

//------------------------------------------------------------------------------
// parse_amd_module
//------------------------------------------------------------------------------
AmdModule parse_amd_module(std::string_view asm_text, std::string_view target,
                           std::string_view source_name) {
  ensure_targets_initialized();

  // Normalize the target. hipcc/clang emit `.amdgcn_target
  // "amdgcn-amd-amdhsa--gfx1101"`; LLVM 21.1's TargetID comparison only
  // succeeds when the CPU is given to the subtarget separately from a bare
  // "amdgcn-amd-amdhsa" triple (see header note). Accept either a CPU name
  // ("gfx1101") or a full triple ("amdgcn-amd-amdhsa--gfx1101").
  const std::string BareTriple = "amdgcn-amd-amdhsa";
  std::string Cpu;
  if (StringRef(target).starts_with("gfx")) {
    Cpu = std::string(target);
  } else {
    const std::string T = std::string(target);
    const std::size_t Pos = T.rfind("--");
    if (Pos != std::string::npos && Pos + 2 < T.size()) {
      Cpu = T.substr(Pos + 2);
    } else {
      Cpu = T; // e.g. plain "gfx1101" or an already-bare triple
    }
  }
  const std::string FullTriple = BareTriple + "--" + Cpu;

  std::string Error;
  Triple TT(Triple::normalize(BareTriple));
  const Target *TheTarget = TargetRegistry::lookupTarget("amdgcn", TT, Error);
  if (TheTarget == nullptr) {
    throw AmdParseError(
        "amdparse: cannot resolve AMDGPU target for '" + FullTriple +
            "': " + Error,
        FullTriple, Error, 0, 0);
  }

  // Source buffer + diagnostic capture.
  SourceMgr SrcMgr;
  DiagCapture Capture;
  SrcMgr.setDiagHandler(diag_handler, &Capture);
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBufferCopy(std::string(asm_text),
                                     std::string(source_name));
  SrcMgr.AddNewSourceBuffer(std::move(Buffer), SMLoc());

  McSession S;
  const MCTargetOptions MCOptions;
  S.MRI.reset(TheTarget->createMCRegInfo(BareTriple));
  if (!S.MRI) {
    throw AmdParseError("amdparse: no register info for '" + FullTriple + "'",
                        FullTriple, "createMCRegInfo failed", 0, 0);
  }
  S.MAI.reset(TheTarget->createMCAsmInfo(*S.MRI, BareTriple, MCOptions));
  if (!S.MAI) {
    throw AmdParseError("amdparse: no asm info for '" + FullTriple + "'",
                        FullTriple, "createMCAsmInfo failed", 0, 0);
  }
  S.STI.reset(TheTarget->createMCSubtargetInfo(BareTriple, Cpu, ""));
  if (!S.STI) {
    throw AmdParseError("amdparse: no subtarget info for '" + FullTriple + "'",
                        FullTriple, "createMCSubtargetInfo failed", 0, 0);
  }
  S.MII.reset(TheTarget->createMCInstrInfo());
  if (!S.MII) {
    throw AmdParseError(
        "amdparse: no instruction info for '" + FullTriple + "'", FullTriple,
        "createMCInstrInfo failed", 0, 0);
  }

  MCContext Ctx(TT, S.MAI.get(), S.MRI.get(), S.STI.get(), &SrcMgr,
                &MCOptions);
  S.MOFI.reset(TheTarget->createMCObjectFileInfo(Ctx, /*PIC=*/false,
                                                 /*LargeCodeModel=*/false));
  Ctx.setObjectFileInfo(S.MOFI.get());

  // Route MCContext-level diagnostics (e.g. symbol redefinition / some parser
  // errors) into the same capture as the SourceMgr handler.
  Ctx.setDiagnosticHandler(
      [&Capture](const SMDiagnostic &Diag, bool, const SourceMgr &,
                 std::vector<const MDNode *> &) {
        if (Diag.getKind() == SourceMgr::DK_Error && !Capture.saw_error) {
          Capture.saw_error = true;
          Capture.message = Diag.getMessage().str();
          Capture.line = static_cast<unsigned>(std::max(Diag.getLineNo(), 0));
          Capture.column =
              static_cast<unsigned>(std::max(Diag.getColumnNo(), 0));
        }
      });

  // Recording streamer + AMDGPU null target streamer (so the target-level
  // directives exercised by hipcc -S output parse cleanly).
  S.IP.reset(TheTarget->createMCInstPrinter(TT, /*SyntaxVariant=*/0, *S.MAI,
                                            *S.MII, *S.MRI));
  auto *RawStreamer =
      new AmdRecordStreamer(Ctx, *S.MRI, SrcMgr, S.IP.get());
  S.Streamer.reset(RawStreamer);
  TheTarget->createNullTargetStreamer(*S.Streamer);

  S.Parser.reset(createMCAsmParser(SrcMgr, Ctx, *S.Streamer, *S.MAI));
  RawStreamer->attachParser(S.Parser.get());
  S.TAP.reset(TheTarget->createMCAsmParser(*S.STI, *S.Parser, *S.MII,
                                           MCOptions));
  if (!S.TAP) {
    throw AmdParseError(
        "amdparse: this target does not support assembly parsing for '" +
            FullTriple + "'",
        FullTriple, "createMCAsmParser (target) failed", 0, 0);
  }
  S.Parser->setTargetParser(*S.TAP);

  bool Failed = S.Parser->Run(/*NoInitialTextSection=*/false);
  if (Failed || Capture.saw_error) {
    std::ostringstream Msg;
    Msg << "amdparse: failed to parse " << source_name << " for target "
        << FullTriple;
    if (Capture.line) {
      Msg << " at line " << Capture.line;
    }
    if (!Capture.message.empty()) {
      Msg << ": " << Capture.message;
    }
    throw AmdParseError(Msg.str(), FullTriple, Capture.message, Capture.line,
                        Capture.column);
  }

  AmdModule Mod;
  Mod.target_triple = FullTriple;
  Mod.target_cpu = Cpu;
  Mod.source_name = std::string(source_name);
  Mod.instructions = RawStreamer->instructions();
  Mod.blocks = build_cfg(Mod.instructions, RawStreamer->events());
  return Mod;
}

AmdModule parse_amd_file(const std::string &path, std::string_view target) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buf =
      MemoryBuffer::getFile(path, /*IsText=*/true);
  if (std::error_code EC = Buf.getError()) {
    const std::string Triple = "amdgcn-amd-amdhsa--" + std::string(target);
    throw AmdParseError("amdparse: cannot open '" + path + "': " +
                            EC.message(),
                        Triple, EC.message(), 0, 0);
  }
  return parse_amd_module((*Buf)->getBuffer(), target, path);
}
} // namespace kernscope
