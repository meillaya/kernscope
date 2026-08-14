//===- driver_flash.cpp - FlashAttention family adapter --------------------===//
//
// kernscope tools/tune todo 28. The FlashAttention-lite adapter for the
// driver pipeline (todo-27 generator). Gate family is "attention" (the golden
// harness name) with shape Nq,Nk,d. The bounded "auto" grid reads the
// generator's choice arrays (Br/Bc {32,64} × stages {1,2} × vec {1,2,4} ×
// padding layout; causal off — the causal dimension is a user flag in the
// generator, not a search axis here).
//
//===----------------------------------------------------------------------===//

#include "kernscope/tune/flash_attention_gen.h"

#include "driver_families.h"
#include "driver_pipeline_impl.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kernscope::tune {
namespace driver_detail {

namespace {

struct FlashTraits {
  static constexpr const char *kFamily = "flash_attention";
  static constexpr const char *kGoldenFamily = "attention";
  static constexpr bool kCausal = false;
  static constexpr roofline::KernelFamily kRooflineFamily =
      roofline::KernelFamily::kFlashAttention;
  static constexpr std::size_t kShapeArity = 3;

  using Config = FlashAttentionConfig;
  using Problem = FlashAttentionProblem;

  static Problem make_problem(const DriverOptions &opts) {
    return make_problem_for(opts.shape, opts.arch);
  }

  static Problem make_problem_for(const std::vector<std::int64_t> &shape,
                                  const std::string &arch) {
    Problem p;
    p.nq = shape[0];
    p.nk = shape[1];
    p.d = shape[2];
    p.arch = arch;
    p.dtype_bytes = 2;  // bf16 (todo-23 golden dtype)
    return p;
  }

  // Gates at the tune shape (online softmax has no long-K drift).
  static std::vector<std::int64_t> gate_shape(const DriverOptions &opts) {
    return opts.gate_shape.empty() ? opts.shape : opts.gate_shape;
  }

  static std::string shape_string(const std::vector<std::int64_t> &s) {
    return std::to_string(s[0]) + "," + std::to_string(s[1]) + "," +
           std::to_string(s[2]);
  }

  static std::vector<Config> enumerate(const DriverOptions &opts) {
    std::vector<Config> grid;
    if (opts.grid_preset == "full") {
      for (const int br : kTileChoices)
        for (const int bc : kTileChoices)
          for (const int st : kStageChoices)
            for (const int v : kVecWidthChoices)
              for (const SmemLayout lay : kSmemLayoutChoices) {
                Config c;
                c.tile_m = br;
                c.tile_n = bc;
                c.stages = st;
                c.vec_width = v;
                c.smem_layout = lay;
                c.causal = false;
                grid.push_back(c);
              }
      return grid;
    }
    // "auto" bounded grid (>= 20 configs).
    for (const int br : kTileChoices)
      for (const int bc : kTileChoices)
        for (const int st : kStageChoices)
          for (const int v : kVecWidthChoices) {
            Config c;
            c.tile_m = br;
            c.tile_n = bc;
            c.stages = st;
            c.vec_width = v;
            c.smem_layout = SmemLayout::kPadding;
            c.causal = false;
            grid.push_back(c);
          }
    return grid;
  }

  static ConfigVerdict validate(const Config &c, const Problem &p) {
    return validate_config(c, p);
  }
  static std::int64_t smem_bytes(const Config &c, const Problem &p) {
    return kernscope::tune::smem_bytes(c, p);
  }
  static LaunchGeometry geometry(const Config &c, const Problem &p) {
    return launch_geometry(c, p);
  }
  static std::string symbol(const Config &c) { return kernel_symbol(c); }
  static bool generate(const Config &c, const Problem &p,
                       const std::string &path, std::string *err) {
    return generate_to_file(c, p, path, err);
  }
  static int block_size(const Config &c, const Problem &p) {
    const LaunchGeometry g = launch_geometry(c, p);
    return g.block_x * g.block_y;
  }
  static int estimated_regs(const Config &, const Problem &) {
    return 0;  // not modeled (register-light scalar template)
  }
  static db::json::Value config_json(const Config &c) {
    return db::json::object(
        {{"tile_m", db::json::Value(c.tile_m)},
         {"tile_n", db::json::Value(c.tile_n)},
         {"stages", db::json::Value(c.stages)},
         {"vec_width", db::json::Value(c.vec_width)},
         {"smem_layout", db::json::Value(smem_layout_name(c.smem_layout))},
         {"causal", db::json::Value(c.causal)}});
  }
  static std::vector<std::string> bench_bins() {
    return {"q.bin", "k.bin", "v.bin"};
  }
  static std::vector<std::int32_t> scalar_args(
      const std::vector<std::int64_t> &s) {
    return {static_cast<std::int32_t>(s[0]),
            static_cast<std::int32_t>(s[1]),
            static_cast<std::int32_t>(s[2])};
  }
  static std::size_t out_bytes(const std::vector<std::int64_t> &s) {
    return static_cast<std::size_t>(s[0] * s[2] * 2);
  }
  static roofline::Shape roofline_shape(const DriverOptions &opts) {
    roofline::Shape s;
    s.n = opts.shape[0];  // sequence length (query rows)
    s.d = opts.shape[2];  // head dim
    s.dtype_bytes = 2;
    return s;
  }
};

}  // namespace

DriverResult tune_flash_attention(const DriverOptions &opts,
                                  const DriverHooks *hooks) {
  return Tuner<FlashTraits>::run(opts, hooks);
}

}  // namespace driver_detail
}  // namespace kernscope::tune
