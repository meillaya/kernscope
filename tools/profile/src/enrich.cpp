//===- enrich.cpp - per-kernel record enrichment (todo 50) -----------------*- C++ -*-===//
//
// kernscope tools/profile (C6 profiler track).
//
// Implementation of include/kernscope/profile/enrich.h. Every parser here is
// deliberately line-based and noexcept: enrichment must never crash the
// tracer's worker thread or the report path — bad tool output degrades to an
// empty/partial enrichment, never a fabricated number.
//
// The occupancy math is ENTIRELY delegated to libs/occupancy
// (kernscope::occupancy::compute_occupancy, todo 11): this file only picks
// the inputs (registersPerThread, static+dynamic shared, block dims) and
// maps the result into EnrichedKernel — the acceptance asserts the
// enrichment's occupancy field EQUALS the libs/occupancy output for the
// same inputs.
//
//===----------------------------------------------------------------------===//

#include "kernscope/profile/enrich.h"

#include "kernscope/arch.h"
#include "kernscope/occupancy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace kernscope::profile {

namespace {

// Runs a command and returns its stdout (trimmed). Returns "" on any failure
// (fork/exec error, non-zero exit). Never throws.
std::string RunToolOutput(const char *const argv[]) {
  // Build the command line with shell quoting of each argument.
  std::string cmd;
  for (int i = 0; argv[i] != nullptr; ++i) {
    if (i > 0) cmd += ' ';
    const std::string arg = argv[i];
    cmd += '\'';
    for (char c : arg) {
      if (c == '\'') {
        cmd += "'\\''";
      } else {
        cmd += c;
      }
    }
    cmd += '\'';
  }
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
  const int rc = ::pclose(pipe);
  if (rc != 0) {
    return "";
  }
  // Trim trailing whitespace/newlines (a whitespace-only result counts as
  // "no output" for the embedded-PTX detection).
  std::size_t end = out.size();
  while (end > 0) {
    const char c = out[end - 1];
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r' && c != '\f' &&
        c != '\v') {
      break;
    }
    --end;
  }
  out.resize(end);
  return out;
}

std::string Trim(const std::string &s) {
  std::size_t b = 0;
  while (b < s.size() && (s[b] == ' ' || s[b] == '\t')) ++b;
  std::size_t e = s.size();
  while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' ||
                   s[e - 1] == '\n')) {
    --e;
  }
  return s.substr(b, e - b);
}

// Extracts the integer after "<key>:" in text like "REG:10 STACK:0". Returns
// -1 when absent (or not parseable).
int ParseResField(const std::string &line, const char *key) {
  const std::size_t pos = line.find(key);
  if (pos == std::string::npos) {
    return -1;
  }
  const std::size_t colon = line.find(':', pos);
  if (colon == std::string::npos) {
    return -1;
  }
  std::size_t i = colon + 1;
  while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
  long value = 0;
  bool any = false;
  for (; i < line.size() && line[i] >= '0' && line[i] <= '9'; ++i) {
    value = (value * 10) + (line[i] - '0');
    any = true;
  }
  return any ? static_cast<int>(value) : -1;
}

} // namespace

EnrichedKernel enrich_kernel(const KernelRecord &record,
                             const EnrichOptions &options) noexcept {
  EnrichedKernel out;
  try {
    out.record = record;

    const int regs = static_cast<int>(record.registers_per_thread);
    const int block_size =
        static_cast<int>(record.block_x * record.block_y * record.block_z);
    const int smem = static_cast<int>(record.static_shared_bytes) +
                     static_cast<int>(record.dynamic_shared_bytes);

    // No resource evidence -> "unknown-resources" (QA contract: never a
    // fabricated 0%). Also covers block_size <= 0 and unknown arch names
    // (get_arch throws std::out_of_range for foreign keys).
    if (regs <= 0 || block_size <= 0 || options.arch_name.empty() ||
        !kernscope::has_arch(options.arch_name)) {
      out.occupancy_unknown = true;
      out.occupancy = -1.0;
      out.limiter.clear();
      return out;
    }

    const kernscope::ArchInfo &arch =
        kernscope::get_arch(options.arch_name); // guarded by has_arch above
    const kernscope::occupancy::OccupancyResult occ =
        kernscope::occupancy::compute_occupancy(arch, regs, smem, block_size,
                                                options.sgprs_per_thread);
    out.occupancy = occ.occupancy;
    out.occupancy_unknown = false;
    out.limiter = kernscope::occupancy::limiter_name(occ.limiter);
    out.blocks_per_sm = occ.blocks_per_sm;
    out.warps_resident = occ.warps_resident;
    out.max_warps_per_sm = occ.max_warps_per_sm;
  } catch (...) {
    return out;  // documented degradation: partial enrichment, never a crash
  }
  return out;
}

void apply_nvidia_res_usage(const std::string &cuobjdump_text,
                            EnrichedKernel &enriched) noexcept {
  // cuobjdump -res-usage layout:
  //   Resource usage:
  //    Common:
  //     GLOBAL:0
  //   Function _Z11tiny_kernelPKfPfi:
  //     REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:372 ...
  std::size_t pos = 0;
  while (pos < cuobjdump_text.size()) {
    const std::size_t nl = cuobjdump_text.find('\n', pos);
    const std::string line =
        cuobjdump_text.substr(pos, nl == std::string::npos ? std::string::npos
                                                           : nl - pos);
    pos = (nl == std::string::npos) ? cuobjdump_text.size() : nl + 1;

    const std::string t = Trim(line);
    constexpr const char *kFunctionPrefix = "Function ";
    if (t.rfind(kFunctionPrefix, 0) != 0) {
      continue;
    }
    const std::string name = Trim(t.substr(std::strlen(kFunctionPrefix)));
    // Strip a trailing ':' from the function name (cuobjdump prints "Function
    // _Z...:").
    std::string function = name;
    if (!function.empty() && function.back() == ':') {
      function.pop_back();
    }
    if (function.empty()) {
      continue;
    }
    // The resource line is the NEXT non-empty line.
    std::size_t rpos = pos;
    std::string res_line;
    while (rpos < cuobjdump_text.size()) {
      const std::size_t rnl = cuobjdump_text.find('\n', rpos);
      const std::string cand = cuobjdump_text.substr(
          rpos, rnl == std::string::npos ? std::string::npos : rnl - rpos);
      rpos = (rnl == std::string::npos) ? cuobjdump_text.size() : rnl + 1;
      if (!Trim(cand).empty()) {
        res_line = cand;
        break;
      }
    }
    FunctionResourceUsage usage;
    usage.function = function;
    usage.reg = ParseResField(res_line, "REG");
    usage.stack = ParseResField(res_line, "STACK");
    usage.shared = ParseResField(res_line, "SHARED");
    usage.local = ParseResField(res_line, "LOCAL");
    enriched.res_usage.push_back(usage);
  }
}

void apply_ptx_lineinfo(const std::string &ptx_text,
                        EnrichedKernel &enriched) noexcept {
  // .file <N> "<path>"   (PTX emits the .file directive at the END of the
  //                       file; .loc references it by number)
  // \t.loc <N> <line> [<col>]   (tab-indented inside the kernel body)
  // The PTX text existing at all is the ptx_available signal.
  enriched.ptx_available = true;

  // Pass 1: collect the .file <N> "<path>" table (PTX puts it at the END of
  // the file, after every .loc that references it — a single sequential scan
  // would resolve nothing).
  std::vector<std::string> files; // index 0 unused (files are 1-based)
  std::size_t pos = 0;
  while (pos < ptx_text.size()) {
    const std::size_t nl = ptx_text.find('\n', pos);
    const std::string line =
        ptx_text.substr(pos, nl == std::string::npos ? std::string::npos
                                                     : nl - pos);
    pos = (nl == std::string::npos) ? ptx_text.size() : nl + 1;

    const std::string t = Trim(line);
    if (t.rfind(".file", 0) != 0 || t.find('"') == std::string::npos) {
      continue;
    }
    // .file <N> "<path>"
    const std::size_t num_start = std::strlen(".file");
    std::size_t i = num_start;
    while (i < t.size() && (t[i] == ' ' || t[i] == '\t')) ++i;
    long num = 0;
    bool num_ok = false;
    while (i < t.size() && t[i] >= '0' && t[i] <= '9') {
      num = (num * 10) + (t[i] - '0');
      num_ok = true;
      ++i;
    }
    const std::size_t q1 = t.find('"', i);
    const std::size_t q2 =
        q1 == std::string::npos ? std::string::npos : t.find('"', q1 + 1);
    if (!num_ok || q1 == std::string::npos || q2 == std::string::npos) {
      continue;
    }
    if (num > 0 && static_cast<std::size_t>(num) >= files.size()) {
      files.resize(static_cast<std::size_t>(num) + 1);
    }
    files[static_cast<std::size_t>(num)] = t.substr(q1 + 1, q2 - q1 - 1);
  }

  // Pass 2: the .loc directives (file table already resolved).
  pos = 0;
  while (pos < ptx_text.size()) {
    const std::size_t nl = ptx_text.find('\n', pos);
    const std::string line =
        ptx_text.substr(pos, nl == std::string::npos ? std::string::npos
                                                     : nl - pos);
    pos = (nl == std::string::npos) ? ptx_text.size() : nl + 1;
    const std::string t = Trim(line);
    if (t.rfind(".loc", 0) == 0) {
      // .loc <file> <line> [<col>]
      std::size_t i = std::strlen(".loc");
      while (i < t.size() && (t[i] == ' ' || t[i] == '\t')) ++i;
      long file_num = 0;
      while (i < t.size() && t[i] >= '0' && t[i] <= '9') {
        file_num = (file_num * 10) + (t[i] - '0');
        ++i;
      }
      while (i < t.size() && (t[i] == ' ' || t[i] == '\t')) ++i;
      long line_num = 0;
      while (i < t.size() && t[i] >= '0' && t[i] <= '9') {
        line_num = (line_num * 10) + (t[i] - '0');
        ++i;
      }
      SourceMappingEntry entry;
      entry.source_line = static_cast<int>(line_num);
      if (file_num > 0 && static_cast<std::size_t>(file_num) < files.size()) {
        entry.source_file = files[static_cast<std::size_t>(file_num)];
      }
      // The instruction that the .loc applies to is the next non-empty,
      // non-.loc/.file line (the first instruction after the directive).
      std::size_t ipos = pos;
      while (ipos < ptx_text.size()) {
        const std::size_t inl = ptx_text.find('\n', ipos);
        const std::string cand = ptx_text.substr(
            ipos, inl == std::string::npos ? std::string::npos : inl - ipos);
        ipos = (inl == std::string::npos) ? ptx_text.size() : inl + 1;
        const std::string it = Trim(cand);
        if (it.empty() || it.rfind(".loc", 0) == 0 ||
            it.rfind(".file", 0) == 0 || it.rfind(".visible", 0) == 0 ||
            it.rfind(".entry", 0) == 0 || it.rfind(".reg", 0) == 0 ||
            it.rfind(".param", 0) == 0 || it.rfind(".shared", 0) == 0 ||
            it.rfind('{', 0) == 0 || it.rfind('}', 0) == 0) {
          continue;
        }
        entry.asm_text = it;
        break;
      }
      enriched.source_mapping.push_back(std::move(entry));
    }
  }
}

void apply_llvm_objdump_lines(const std::string &objdump_text,
                              EnrichedKernel &enriched) noexcept {
  // llvm-objdump -l output interleaves source comments with the disassembly:
  //   ; /path/to/file.cu:3
  //   \ts_load_b32 s3, s[0:1], 0x10    // bytes <addr>
  std::size_t pos = 0;
  while (pos < objdump_text.size()) {
    const std::size_t nl = objdump_text.find('\n', pos);
    const std::string line =
        objdump_text.substr(pos, nl == std::string::npos ? std::string::npos
                                                         : nl - pos);
    pos = (nl == std::string::npos) ? objdump_text.size() : nl + 1;

    const std::string t = Trim(line);
    if (t.rfind("; ", 0) != 0) {
      continue;
    }
    // "; /path/to/file.cu:3" — split at the LAST ':' that is followed by
    // digits (the line number; Windows paths could contain ':' in the drive
    // letter, so scan from the end).
    const std::size_t colon = t.rfind(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::string file = Trim(t.substr(2, colon - 2));
    const std::string line_num = Trim(t.substr(colon + 1));
    long value = 0;
    bool ok = !line_num.empty();
    for (char c : line_num) {
      if (c < '0' || c > '9') {
        ok = false;
        break;
      }
      value = (value * 10) + (c - '0');
    }
    if (!ok || file.empty()) {
      continue;
    }
    // The ISA instruction is the next line starting with a tab.
    std::size_t ipos = pos;
    std::string asm_text;
    while (ipos < objdump_text.size()) {
      const std::size_t inl = objdump_text.find('\n', ipos);
      const std::string cand = objdump_text.substr(
          ipos, inl == std::string::npos ? std::string::npos : inl - ipos);
      ipos = (inl == std::string::npos) ? objdump_text.size() : inl + 1;
      if (cand.empty()) {
        continue;
      }
      if (cand[0] == '\t') {
        asm_text = Trim(cand);
        break;
      }
      if (cand.rfind("; ", 0) == 0) {
        break; // next source comment without an instruction between
      }
    }
    SourceMappingEntry entry;
    entry.source_file = std::move(file);
    entry.source_line = static_cast<int>(value);
    entry.asm_text = std::move(asm_text);
    enriched.source_mapping.push_back(std::move(entry));
  }
}

bool module_has_embedded_ptx(const std::string &cubin_path) noexcept {
  // cubin-only modules carry NO embedded PTX: `cuobjdump -ptx` on them exits
  // 0 with EMPTY output (verified on nvcc 12.6 cubins). Non-empty output ==
  // embedded PTX present. cuobjdump absent / command failure -> false (the
  // degradation direction — we never fabricate PTX).
  const char *const argv[] = {"cuobjdump", "-ptx", cubin_path.c_str(),
                              nullptr};
  return !RunToolOutput(argv).empty();
}

std::string enriched_kernel_to_json(const char *vendor,
                                    const EnrichedKernel &enriched) {
  // The todo-49 record JSON + the enrichment fields. occupancy is a JSON
  // number (fraction 0..1) or the string "unknown-resources".
  const KernelRecord &r = enriched.record;
  std::string occupancy_field =
      enriched.occupancy_unknown
          ? "\"unknown-resources\""
          : (std::to_string(enriched.occupancy)); // JSON number

  char buf[2048];
  const int n = std::snprintf(
      buf, sizeof(buf),
      "{\"vendor\":\"%s\",\"kind\":\"kernel\",\"name\":\"%s\","
      "\"start_ns\":%llu,\"end_ns\":%llu,\"duration_ns\":%llu,"
      "\"grid\":[%lld,%lld,%lld],\"workgroup\":[%lld,%lld,%lld],"
      "\"static_shared_bytes\":%d,\"dynamic_shared_bytes\":%d,"
      "\"shared_memory_executed\":%u,\"registers_per_thread\":%u,"
      "\"local_memory_per_thread\":%u,\"local_memory_total\":%llu,"
      "\"correlation_id\":%u,\"device_id\":%u,\"context_id\":%u,"
      "\"stream_id\":%u,\"grid_id\":%lld,\"launch_type\":%u,"
      "\"is_device_launched\":%u,"
      "\"occupancy\":%s,\"limiter\":\"%s\",\"blocks_per_sm\":%d,"
      "\"warps_resident\":%d,\"max_warps_per_sm\":%d,"
      "\"vgpr\":%d,\"sgpr\":%d,\"accum_vgpr\":%d,\"lds_bytes\":%d,"
      "\"scratch_bytes\":%d,\"kernel_address\":%llu,"
      "\"ptx_available\":%s,\"source_mapping_entries\":%zu,"
      "\"res_usage_functions\":%zu,\"ptx_diagnostics\":%zu}",
      vendor, json_escape(r.kernel_name).c_str(),
      static_cast<unsigned long long>(r.start_ns),
      static_cast<unsigned long long>(r.end_ns),
      static_cast<unsigned long long>(r.duration_ns),
      static_cast<long long>(r.grid_x), static_cast<long long>(r.grid_y),
      static_cast<long long>(r.grid_z), static_cast<long long>(r.block_x),
      static_cast<long long>(r.block_y), static_cast<long long>(r.block_z),
      r.static_shared_bytes, r.dynamic_shared_bytes,
      r.shared_memory_executed, r.registers_per_thread,
      r.local_memory_per_thread,
      static_cast<unsigned long long>(r.local_memory_total),
      r.correlation_id, r.device_id, r.context_id, r.stream_id,
      static_cast<long long>(r.grid_id), r.launch_type,
      r.is_device_launched, occupancy_field.c_str(), json_escape(enriched.limiter).c_str(),
      enriched.blocks_per_sm, enriched.warps_resident, enriched.max_warps_per_sm,
      enriched.arch_vgpr_count, enriched.sgpr_count, enriched.accum_vgpr_count,
      enriched.group_segment_bytes, enriched.private_segment_bytes,
      static_cast<unsigned long long>(enriched.kernel_address),
      enriched.ptx_available ? "true" : "false", enriched.source_mapping.size(),
      enriched.res_usage.size(), enriched.ptx_diagnostics.size());
  return (n > 0) ? std::string(buf, static_cast<std::size_t>(n)) : std::string();
}

} // namespace kernscope::profile
