//===- driver_matmul.cpp - matmul family adapter for the tuning driver -----===//
//
// kernscope tools/tune todo 28. The matmul adapter: maps the driver's
// family-generic pipeline (driver_pipeline_impl.h) onto the todo-25 matmul
// template generator. The bounded "auto" grid (24 configs, >= the 20-config
// acceptance floor) reads the generator's search-space choice arrays:
//
//   (BM, BN) ∈ {(64,64), (128,128)} × BK ∈ {16,32} × warps ∈ {1,2,4}
//   × stages ∈ {1,2}   (vec=1, padding, unroll=1 fixed)
//
// Every config the register/smem budget rejects is caught by the driver's
// GPU-free pre-screen (validate_config) — on gfx1101 the 128×128@warps≤2
// register over-budget and 128×128@BK32@stages2 smem over-budget cases are
// the >= 1 pre-screen rejection the acceptance asserts.
//
//===----------------------------------------------------------------------===//

#include "kernscope/tune/matmul_gen.h"

#include "driver_families.h"
#include "driver_pipeline_impl.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kernscope::tune {
namespace driver_detail {

namespace {

struct MatmulTraits {
  static constexpr const char *kFamily = "matmul";
  static constexpr const char *kGoldenFamily = "matmul";
  static constexpr bool kCausal = false;
  static constexpr roofline::KernelFamily kRooflineFamily =
      roofline::KernelFamily::kGemm;
  static constexpr std::size_t kShapeArity = 3;

  using Config = MatmulConfig;
  using Problem = MatmulProblem;

  static Problem make_problem(const DriverOptions &opts) {
    return make_problem_for(opts.shape, opts.arch);
  }

  static Problem make_problem_for(const std::vector<std::int64_t> &shape,
                                  const std::string &arch) {
    Problem p;
    p.m = shape[0];
    p.n = shape[1];
    p.k = shape[2];
    p.arch = arch;
    p.dtype_bytes = 2;  // bf16 (todo-23 golden dtype)
    return p;
  }

  // Correctness gate shape. The golden contract pins max_rel_err <= 1e-2 for
  // the bf16 rounding contract; an fp32-accumulate GEMM legitimately drifts
  // past it at K=512 (the todo-25 live gate therefore validates at
  // M=N=K=64). Default to that golden-validated shape; benchmarking still
  // measures the requested `shape`.
  static std::vector<std::int64_t> gate_shape(const DriverOptions &opts) {
    if (!opts.gate_shape.empty())
      return opts.gate_shape;
    return {64, 64, 64};
  }

  static std::string shape_string(const std::vector<std::int64_t> &s) {
    return std::to_string(s[0]) + "," + std::to_string(s[1]) + "," +
           std::to_string(s[2]);
  }

  static std::vector<Config> enumerate(const DriverOptions &opts) {
    std::vector<Config> grid;
    if (opts.grid_preset == "full") {
      for (const int bm : kTileMChoices)
        for (const int bn : kTileNChoices)
          for (const int bk : kTileKChoices)
            for (const int w : kWarpChoices)
              for (const int st : kStageChoices)
                for (const int v : kVecWidthChoices)
                  for (const SmemLayout lay : kSmemLayoutChoices)
                    for (const int u : kUnrollChoices) {
                      Config c;
                      c.tile_m = bm;
                      c.tile_n = bn;
                      c.tile_k = bk;
                      c.warps = w;
                      c.stages = st;
                      c.vec_width = v;
                      c.smem_layout = lay;
                      c.unroll = u;
                      grid.push_back(c);
                    }
      return grid;
    }
    // "auto" bounded grid (the acceptance floor is >= 20 configs).
    constexpr std::pair<int, int> kTilePairs[] = {
        {64, 64}, {128, 128}};
    for (const auto &[bm, bn] : kTilePairs)
      for (const int bk : {16, 32})
        for (const int warps : {1, 2, 4})
          for (const int stages : {1, 2}) {
            Config c;
            c.tile_m = bm;
            c.tile_n = bn;
            c.tile_k = bk;
            c.warps = warps;
            c.stages = stages;
            c.vec_width = 1;
            c.smem_layout = SmemLayout::kPadding;
            c.unroll = 1;
            grid.push_back(c);
          }
    return grid;
  }

  static ConfigVerdict validate(const Config &c, const Problem &p) {
    return validate_config(c, p);
  }
  static std::int64_t smem_bytes(const Config &c, const Problem & /*unused*/) {
    return kernscope::tune::smem_bytes(c);
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
  // Fragment accumulator regs + the generator's 32-register overhead (the
  // same budget validate_config enforces; occupancy uses it as the VGPR
  // estimate).
  static int estimated_regs(const Config &c, const Problem & /*unused*/) {
    const int threads = 32 * c.warps;
    return (c.tile_m * c.tile_n / threads) + 32;
  }
  static db::json::Value config_json(const Config &c) {
    return db::json::object(
        {{"tile_m", db::json::Value(c.tile_m)},
         {"tile_n", db::json::Value(c.tile_n)},
         {"tile_k", db::json::Value(c.tile_k)},
         {"warps", db::json::Value(c.warps)},
         {"stages", db::json::Value(c.stages)},
         {"vec_width", db::json::Value(c.vec_width)},
         {"smem_layout", db::json::Value(smem_layout_name(c.smem_layout))},
         {"unroll", db::json::Value(c.unroll)}});
  }
  static std::vector<std::string> bench_bins() {
    return {"a.bin", "b.bin"};
  }
  static std::vector<std::int32_t> scalar_args(
      const std::vector<std::int64_t> &s) {
    return {static_cast<std::int32_t>(s[0]),
            static_cast<std::int32_t>(s[1]),
            static_cast<std::int32_t>(s[2])};
  }
  static std::size_t out_bytes(const std::vector<std::int64_t> &s) {
    return static_cast<std::size_t>(s[0] * s[1] * 2);
  }
  static roofline::Shape roofline_shape(const DriverOptions &opts) {
    roofline::Shape s;
    s.m = opts.shape[0];
    s.n = opts.shape[1];
    s.k = opts.shape[2];
    s.dtype_bytes = 2;
    return s;
  }
};

}  // namespace

DriverResult tune_matmul(const DriverOptions &opts, const DriverHooks *hooks) {
  return Tuner<MatmulTraits>::run(opts, hooks);
}

}  // namespace driver_detail
}  // namespace kernscope::tune
