//===- profile_cli_test.cpp - kernscope-profile CLI unit suite (todo 52) ---*- C++ -*-===//
//
// The todo-52 no-gpu acceptance units for the CLI pipeline (cli_pipeline.h):
//   * JsonlParsingRoundTrip — the enriched JSONL shape the todo-49/50 tracer
//     emits parses back into the EnrichedKernel model (name/duration/occupancy/
//     limiter/vgpr/LDS + the "unknown-resources" degradation marker);
//   * NoKernelsNegativeIsExit0Report — the QA negative: a run with zero
//     kernel records yields no_kernels_traced == true, the text report says
//     exactly "no kernels traced" (never an error);
//   * ReportJsonCarriesNumericFields — build_report on a synthetic trace
//     (layernorm family) yields ranking + per-kernel hints with evidence, and
//     report_to_json round-trips through libs/db JSON with machine-readable
//     numbers (duration_ns > 0, occupancy, limiter, evidence keys);
//   * OwnerSectionBTextShape — report_to_text prints the owner's example
//     SHAPE (Kernel:/Time:/Registers/thread:/Shared memory/block:/Likely
//     limiter:/Compiler hint:) with locally-measured values;
//   * HtmlReportUsesLibsReportEmbeddedJson — the --html path builds a
//     libs/report KernelReport and kernscope::to_html embeds a JSON payload
//     that parses (tests/report/check_html.py extraction contract).
//
// No GPU required (no-gpu label): the pipeline's report stages are pure
// host-side over the tracer's JSONL.
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>

#include "cli_pipeline.h"
#include "kernscope/db.h"
#include "kernscope/profile/enrich.h"
#include "kernscope/report.h"

namespace {

using kernscope::profile::cli::build_report;
using kernscope::profile::cli::ProfileReport;
using kernscope::profile::cli::TraceOptions;
using kernscope::profile::EnrichedKernel;

// One enriched kernel record in the tracer's JSONL shape (todo 50 emitter).
std::string EnrichedJsonl(const char *name, std::uint64_t duration_ns,
                          double occupancy, const char *limiter,
                          int vgpr, int lds) {
  return std::string("{\"vendor\":\"amd\",\"kind\":\"kernel\",\"name\":\"") +
         name +
         "\",\"start_ns\":0,\"end_ns\":0,\"duration_ns\":" +
         std::to_string(duration_ns) + ",\"grid\":[1024,1,1],"
         "\"workgroup\":[128,1,1],\"static_shared_bytes\":" +
         std::to_string(lds) + ",\"dynamic_shared_bytes\":0,"
         "\"shared_memory_executed\":0,\"registers_per_thread\":" +
         std::to_string(vgpr) +
         ",\"local_memory_per_thread\":0,\"local_memory_total\":0,"
         "\"correlation_id\":7,\"device_id\":0,\"context_id\":0,"
         "\"stream_id\":0,\"grid_id\":1,\"launch_type\":0,"
         "\"is_device_launched\":0,\"occupancy\":" +
         std::to_string(occupancy) + ",\"limiter\":\"" + limiter +
         "\",\"blocks_per_sm\":8,\"warps_resident\":16,"
         "\"max_warps_per_sm\":16,\"vgpr\":" + std::to_string(vgpr) +
         ",\"sgpr\":128,\"accum_vgpr\":0,\"lds_bytes\":" +
         std::to_string(lds) + ",\"scratch_bytes\":0,"
         "\"kernel_address\":12345,\"ptx_available\":false,"
         "\"source_mapping_entries\":0,\"res_usage_functions\":0,"
         "\"ptx_diagnostics\":0}";
}

TraceOptions MakeOptions() {
  TraceOptions o;
  o.target = "layernorm_app";
  o.vendor = "hip";
  o.arch_name = "sm_80"; // deterministic roofline peaks for the unit trace
  return o;
}

} // namespace

//--- JSONL parsing -------------------------------------------------------------//

TEST(ProfileCli, JsonlParsingRoundTrip) {
  const std::string jsonl =
      EnrichedJsonl("_Z15layernorm_fusedPKtS0_S0_Ptfi.kd", 41200, 1.0,
                    "registers", 24, 1024) +
      "\n{\"vendor\":\"amd\",\"kind\":\"api\",\"operation\":\"hipLaunchKernel"
      "\",\"correlation_id\":7,\"start_ns\":0,\"end_ns\":0,\"thread_id\":0}\n"
      "{\"vendor\":\"amd\",\"kind\":\"kernel\",\"name\":\"other.kd\","
      "\"duration_ns\":1000,\"grid\":[1,1,1],\"workgroup\":[1,1,1],"
      "\"occupancy\":\"unknown-resources\",\"limiter\":\"\"}\n"
      "this line is not json and must be skipped\n";

  std::vector<EnrichedKernel> kernels;
  std::size_t api = 0;
  kernscope::profile::cli::parse_profile_jsonl(jsonl, &kernels, &api);

  ASSERT_EQ(kernels.size(), 2u);
  EXPECT_EQ(api, 1u);
  const EnrichedKernel &layernorm = kernels[0];
  EXPECT_EQ(layernorm.record.kernel_name,
            "_Z15layernorm_fusedPKtS0_S0_Ptfi.kd");
  EXPECT_EQ(layernorm.record.duration_ns, 41200u);
  EXPECT_EQ(layernorm.record.grid_x, 1024);
  EXPECT_EQ(layernorm.record.block_x, 128);
  EXPECT_EQ(layernorm.record.registers_per_thread, 24);
  EXPECT_EQ(layernorm.record.static_shared_bytes, 1024);
  EXPECT_DOUBLE_EQ(layernorm.occupancy, 1.0);
  EXPECT_FALSE(layernorm.occupancy_unknown);
  EXPECT_EQ(layernorm.limiter, "registers");
  EXPECT_EQ(layernorm.arch_vgpr_count, 24);
  EXPECT_EQ(layernorm.group_segment_bytes, 1024);
  EXPECT_FALSE(layernorm.ptx_available);

  // The degradation marker: "unknown-resources" -> occupancy_unknown == true.
  EXPECT_TRUE(kernels[1].occupancy_unknown);
  EXPECT_EQ(kernels[1].occupancy, -1.0);
}

//--- QA negative: no kernels traced --------------------------------------------//

TEST(ProfileCli, NoKernelsNegativeIsExit0Report) {
  std::vector<EnrichedKernel> kernels;
  std::size_t api = 0;
  kernscope::profile::cli::parse_profile_jsonl(
      "{\"vendor\":\"amd\",\"kind\":\"api\",\"operation\":\"hipMalloc\","
      "\"correlation_id\":1}\n",
      &kernels, &api);
  ASSERT_TRUE(kernels.empty());
  EXPECT_EQ(api, 1u);

  const ProfileReport report = build_report(kernels, MakeOptions());
  EXPECT_TRUE(report.no_kernels_traced);
  EXPECT_TRUE(report.kernels.empty());
  EXPECT_TRUE(report.ranking.empty());

  const std::string text = kernscope::profile::cli::report_to_text(report);
  EXPECT_EQ(text, "no kernels traced\n");

  // The JSON report carries the negative signal (exit 0, not an error).
  const std::string json = kernscope::profile::cli::report_to_json(report);
  std::string error;
  const std::optional<kernscope::db::json::Value> parsed =
      kernscope::db::json::Value::parse(json, error);
  if (!parsed.has_value()) {
    FAIL() << error;
    return;
  }
  EXPECT_TRUE(parsed.value().find("no_kernels_traced")->as_bool());
  EXPECT_EQ(parsed.value().find("kernel_count")->as_number(), 0.0);
}

//--- report assembly + JSON -----------------------------------------------------//

TEST(ProfileCli, ReportJsonCarriesNumericFields) {
  std::vector<EnrichedKernel> kernels;
  kernscope::profile::cli::parse_profile_jsonl(
      EnrichedJsonl("_Z15layernorm_fusedPKtS0_S0_Ptfi.kd", 41200, 1.0,
                    "registers", 24, 1024) +
          "\n",
      &kernels, nullptr);
  ASSERT_EQ(kernels.size(), 1u);

  const ProfileReport report = build_report(kernels, MakeOptions());
  EXPECT_FALSE(report.no_kernels_traced);
  ASSERT_EQ(report.ranking.size(), 1u);
  EXPECT_EQ(report.ranking[0].kernel_name,
            "_Z15layernorm_fusedPKtS0_S0_Ptfi.kd");
  EXPECT_EQ(report.ranking[0].launches, 1u);
  EXPECT_EQ(report.ranking[0].total_ns, 41200u);
  EXPECT_EQ(report.wall_time_ns, 41200u);
  ASSERT_EQ(report.diagnostics.size(), 1u);
  // The layernorm family is KNOWN to the rules engine (todo 51): with full
  // occupancy (1.0) the roofline classification fires -> >= 1 hint with
  // evidence, never a fabricated hint.
  ASSERT_FALSE(report.diagnostics[0].hints.empty());
  for (const kernscope::Finding &f : report.diagnostics[0].hints) {
    EXPECT_FALSE(f.evidence.empty());
  }

  const std::string json = kernscope::profile::cli::report_to_json(report);
  std::string error;
  const std::optional<kernscope::db::json::Value> parsed =
      kernscope::db::json::Value::parse(json, error);
  if (!parsed.has_value()) {
    ADD_FAILURE() << error;
    return;
  }
  const kernscope::db::json::Value *kernels_arr =
      parsed.value().find("kernels");
  if (kernels_arr == nullptr) {
    FAIL() << "kernels array missing";
    return;
  }
  ASSERT_EQ(kernels_arr->as_array().size(), 1u);
  const kernscope::db::json::Value &k = kernels_arr->as_array()[0];
  EXPECT_EQ(k.find("kernel_name")->as_string(),
            "_Z15layernorm_fusedPKtS0_S0_Ptfi.kd");
  EXPECT_GT(k.find("duration_ns")->as_number(), 0.0);
  EXPECT_GT(k.find("occupancy")->as_number(), 0.0);
  EXPECT_LE(k.find("occupancy")->as_number(), 1.0);
  EXPECT_EQ(k.find("limiter")->as_string(), "registers");
  const kernscope::db::json::Value *hints = k.find("hints");
  ASSERT_NE(hints, nullptr);
  ASSERT_GE(hints->as_array().size(), 1u);
  const kernscope::db::json::Value &hint = hints->as_array()[0];
  EXPECT_FALSE(hint.find("id")->as_string().empty());
  const kernscope::db::json::Value *evidence = hint.find("evidence");
  ASSERT_NE(evidence, nullptr);
  EXPECT_FALSE(evidence->as_object().empty());
}

//--- owner Section B text shape -------------------------------------------------//

TEST(ProfileCli, OwnerSectionBTextShape) {
  std::vector<EnrichedKernel> kernels;
  kernscope::profile::cli::parse_profile_jsonl(
      EnrichedJsonl("_Z15layernorm_fusedPKtS0_S0_Ptfi.kd", 41200, 1.0,
                    "registers", 24, 1024) +
          "\n",
      &kernels, nullptr);
  const ProfileReport report = build_report(kernels, MakeOptions());
  const std::string text = kernscope::profile::cli::report_to_text(report);

  // The owner's example SHAPE (docs/research/owner-examples.md Section B)
  // with locally-measured values — all six fields present.
  EXPECT_NE(text.find("Kernel: "), std::string::npos);
  EXPECT_NE(text.find("Time: "), std::string::npos);
  EXPECT_NE(text.find("Registers/thread: 24"), std::string::npos);
  EXPECT_NE(text.find("Shared memory/block: 1 KB"), std::string::npos);
  EXPECT_NE(text.find("Likely limiter: "), std::string::npos);
  EXPECT_NE(text.find("Compiler hint:"), std::string::npos);
}

//--- libs/report HTML path ------------------------------------------------------//

TEST(ProfileCli, HtmlReportUsesLibsReportEmbeddedJson) {
  std::vector<EnrichedKernel> kernels;
  kernscope::profile::cli::parse_profile_jsonl(
      EnrichedJsonl("_Z15layernorm_fusedPKtS0_S0_Ptfi.kd", 41200, 1.0,
                    "registers", 24, 1024) +
          "\n",
      &kernels, nullptr);
  const ProfileReport report = build_report(kernels, MakeOptions());

  const kernscope::KernelReport model =
      kernscope::profile::cli::build_kernel_report_model(report, 0);
  const std::string html = kernscope::to_html(model); // must not throw

  // The embedded payload parses (tests/report/check_html.py contract): the
  // byte-exact text between the report-data script tags is valid JSON.
  const std::string open =
      "<script id=\"report-data\" type=\"application/json\">";
  const std::string close = "</script>";
  const std::size_t begin = html.find(open);
  ASSERT_NE(begin, std::string::npos);
  const std::size_t payload_begin = begin + open.size();
  const std::size_t end = html.find(close, payload_begin);
  ASSERT_NE(end, std::string::npos);
  std::string error;
  const std::optional<kernscope::db::json::Value> payload =
      kernscope::db::json::Value::parse(
          html.substr(payload_begin, end - payload_begin), error);
  if (!payload.has_value()) {
    FAIL() << error;
    return;
  }
  EXPECT_EQ(payload.value().find("kernel")->find("name")->as_string(),
            "_Z15layernorm_fusedPKtS0_S0_Ptfi.kd");
}
