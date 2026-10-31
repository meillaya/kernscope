//===- fixture_main.cpp - sample report generator for the HTML gate -*- C++ -*-===//
//
// Emits build/test_reports/{sample.html, sample.json} from the owner's
// fused_attention example model (docs/report/kernscope-technical-report.pdf Section A).
// The sample.json is the "expected input JSON"; check_html.py
// (tests/report/check_html.py) parses sample.html with python3 html.parser,
// extracts the embedded JSON payload and compares it against sample.json.
//
// Usage: kernscope-report-fixture <output-directory>
//
//===----------------------------------------------------------------------===//

#include "kernscope/report.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

kernscope::KernelReport SampleReport() {
  kernscope::KernelReport report;
  report.kernel_name = "fused_attention.ptx";
  report.metrics = {
      {"occupancy", 0.375},
      {"registers_per_thread", 112},
      {"blocks_per_sm", 9},
      {"warps_resident_per_sm", 18},
      {"max_warps_per_sm", 48},
      {"shared_memory_bytes", 67584},
      {"spill_stores", 0},
      {"spill_loads", 0},
  };
  report.limiter = "112 registers/thread";

  report.warnings = {
      {"bank_conflict", "warning",
       {{"text", "shared memory stride likely causes 4-way bank conflicts"},
        {"stride_words", "68"},
        {"way", "4"}}},
      {"divergent_inner_loop", "warning",
       {{"text", "divergent branch inside inner loop"},
        {"loop", "score j-loop"},
        {"predicate", "lane % 8 == j % 8"}}},
      {"coalescing", "warning",
       {{"text", "global loads not 128B coalesced"},
        {"efficiency", "0.5"},
        {"vectorized_ld_global", "0"}}},
  };
  report.suggestions = {
      {"tile_k_32", "suggestion",
       {{"text", "tile K dimension by 32"},
        {"current_tile_k", "64"},
        {"suggested_tile_k", "32"}}},
      {"split_accumulator", "suggestion",
       {{"text", "split accumulator fragment"},
        {"accumulator_elements", "64"},
        {"suggested_fragment_elements", "16"}}},
      {"hoist_predicate", "suggestion",
       {{"text", "move predicate outside loop"},
        {"current_branch_sites", "8"},
        {"suggested_branch_sites", "1"}}},
  };
  return report;
}

bool WriteFile(const std::string &path, const std::string &content) {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    return false;
  out << content;
  return static_cast<bool>(out);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: kernscope-report-fixture <output-directory>\n");
    return 2;
  }
  const std::string dir = argv[1];
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    std::fprintf(stderr, "kernscope-report-fixture: cannot create %s: %s\n",
                 dir.c_str(), ec.message().c_str());
    return 1;
  }
  const kernscope::KernelReport report = SampleReport();

  if (!WriteFile(dir + "/sample.json", kernscope::to_json(report))) {
    std::fprintf(stderr, "kernscope-report-fixture: cannot write %s\n",
                 (dir + "/sample.json").c_str());
    return 1;
  }
  if (!WriteFile(dir + "/sample.html", kernscope::to_html(report))) {
    std::fprintf(stderr, "kernscope-report-fixture: cannot write %s\n",
                 (dir + "/sample.html").c_str());
    return 1;
  }
  std::printf("kernscope-report-fixture: wrote %s and %s\n",
              (dir + "/sample.json").c_str(), (dir + "/sample.html").c_str());
  return 0;
}
