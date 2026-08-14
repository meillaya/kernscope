//===- driver_elem.cpp - elementwise/reduction family adapters -------------===//
//
// kernscope tools/tune todo 28. The four elementwise/reduction adapters
// (bias_gelu, softmax, layernorm, reduction) for the driver pipeline. All
// four share the ElemProblem shape contract (rows, cols) and differ only in
// their config struct, gate family, input-buffer layout and the per-family
// grid (the generators' shared choice arrays kElemBlockChoices /
// kElemVecWidthChoices, plus the per-family mode dimension).
//
// Occupancy pre-screen for these families: regs and smem are NOT modeled
// (the kernels are register-light with small static smem), so the occupancy
// engine limits on threads/blocks only — the pre-screen still rejects a
// config whose block cannot be resident, but the binding limiter is the
// generator's validate_config (shape divisibility, block cap, smem-staged-row
// budget for twopass layernorm).
//
//===----------------------------------------------------------------------===//

#include "kernscope/tune/bias_gelu_gen.h"
#include "kernscope/tune/layernorm_gen.h"
#include "kernscope/tune/reduction_gen.h"
#include "kernscope/tune/softmax_gen.h"

#include "driver_families.h"
#include "driver_pipeline_impl.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kernscope::tune {
namespace driver_detail {

namespace {

// --- shared elementwise adapter plumbing ------------------------------------

struct ElemCommon {
  static ElemProblem make_problem_for(const std::vector<std::int64_t> &shape,
                                      const std::string &arch) {
    ElemProblem p;
    p.rows = shape[0];
    p.cols = shape[1];
    p.arch = arch;
    p.dtype_bytes = 2;  // bf16 (todo-23 golden dtype)
    return p;
  }
  static ElemProblem make_problem(const DriverOptions &opts) {
    return make_problem_for(opts.shape, opts.arch);
  }
  static std::string shape_string(const std::vector<std::int64_t> &s) {
    return std::to_string(s[0]) + "," + std::to_string(s[1]);
  }
  // These families gate at the tune shape (no long-K drift): the golden
  // contract passes at any shape satisfying the generator's divisibility.
  static std::vector<std::int64_t> gate_shape(const DriverOptions &opts) {
    return opts.gate_shape.empty() ? opts.shape : opts.gate_shape;
  }
  static int block_size(const int block) { return block; }
  static int estimated_regs() { return 0; }  // not modeled (see header note)
  static std::int64_t smem_bytes() { return 0; }
  static std::vector<std::int32_t> row_col_args(
      const std::vector<std::int64_t> &s) {
    return {static_cast<std::int32_t>(s[0]),
            static_cast<std::int32_t>(s[1])};
  }
  static std::size_t out_bytes(const std::vector<std::int64_t> &s) {
    return static_cast<std::size_t>(s[0] * s[1] * 2);
  }
  static roofline::Shape roofline_shape(const DriverOptions &opts) {
    roofline::Shape s;
    s.n = opts.shape[0] * opts.shape[1];  // total elements
    s.dtype_bytes = 2;
    return s;
  }
  static db::json::Value config_json(int block, int vec, const char *mode) {
    return db::json::object(
        {{"block", db::json::Value(block)},
         {"vec_width", db::json::Value(vec)},
         {"mode", db::json::Value(mode)}});
  }
};

// --- bias_gelu ---------------------------------------------------------------

struct BiasGeluTraits {
  static constexpr const char *kFamily = "bias_gelu";
  static constexpr const char *kGoldenFamily = "bias_gelu";
  static constexpr bool kCausal = false;
  static constexpr roofline::KernelFamily kRooflineFamily =
      roofline::KernelFamily::kFusedBiasGelu;
  static constexpr std::size_t kShapeArity = 2;

  using Config = BiasGeluConfig;
  using Problem = ElemProblem;

  static Problem make_problem(const DriverOptions &o) {
    return ElemCommon::make_problem(o);
  }
  static Problem make_problem_for(const std::vector<std::int64_t> &shape,
                                  const std::string &arch) {
    return ElemCommon::make_problem_for(shape, arch);
  }
  static std::string shape_string(const std::vector<std::int64_t> &s) {
    return ElemCommon::shape_string(s);
  }
  static std::vector<Config> enumerate(const DriverOptions &opts) {
    std::vector<Config> grid;
    for (const int b : kElemBlockChoices)
      for (const int v : kElemVecWidthChoices) {
        Config c;
        c.block = b;
        c.vec_width = v;
        grid.push_back(c);
      }
    (void)opts;
    return grid;
  }
  static ConfigVerdict validate(const Config &c, const Problem &p) {
    return validate_config(c, p);
  }
  static std::int64_t smem_bytes(const Config &, const Problem &) {
    return ElemCommon::smem_bytes();
  }
  static LaunchGeometry geometry(const Config &c, const Problem &p) {
    return launch_geometry(c, p);
  }
  static std::string symbol(const Config &c) { return kernel_symbol(c); }
  static bool generate(const Config &c, const Problem &p,
                       const std::string &path, std::string *err) {
    return generate_to_file(c, p, path, err);
  }
  static int block_size(const Config &c, const Problem &) { return c.block; }
  static int estimated_regs(const Config &, const Problem &) {
    return ElemCommon::estimated_regs();
  }
  static db::json::Value config_json(const Config &c) {
    return ElemCommon::config_json(c.block, c.vec_width, "-");
  }
  static std::vector<std::string> bench_bins() {
    return {"x.bin", "bias.bin"};
  }
  static std::vector<std::int32_t> scalar_args(
      const std::vector<std::int64_t> &s) {
    // kernel: (x, bias, out, int total, int cols)
    return {static_cast<std::int32_t>(s[0] * s[1]),
            static_cast<std::int32_t>(s[1])};
  }
  static std::size_t out_bytes(const std::vector<std::int64_t> &s) {
    return ElemCommon::out_bytes(s);
  }
  static std::vector<std::int64_t> gate_shape(const DriverOptions &o) {
    return ElemCommon::gate_shape(o);
  }
  static roofline::Shape roofline_shape(const DriverOptions &o) {
    return ElemCommon::roofline_shape(o);
  }
};

// --- softmax -----------------------------------------------------------------

struct SoftmaxTraits {
  static constexpr const char *kFamily = "softmax";
  static constexpr const char *kGoldenFamily = "softmax";
  static constexpr bool kCausal = false;
  static constexpr roofline::KernelFamily kRooflineFamily =
      roofline::KernelFamily::kSoftmax;
  static constexpr std::size_t kShapeArity = 2;

  using Config = SoftmaxConfig;
  using Problem = ElemProblem;

  static Problem make_problem(const DriverOptions &o) {
    return ElemCommon::make_problem(o);
  }
  static Problem make_problem_for(const std::vector<std::int64_t> &shape,
                                  const std::string &arch) {
    return ElemCommon::make_problem_for(shape, arch);
  }
  static std::string shape_string(const std::vector<std::int64_t> &s) {
    return ElemCommon::shape_string(s);
  }
  static std::vector<Config> enumerate(const DriverOptions &opts) {
    std::vector<Config> grid;
    for (const int b : kElemBlockChoices)
      for (const int v : kElemVecWidthChoices)
        for (const ElemMode m : {ElemMode::kOnePass, ElemMode::kTwoPass}) {
          Config c;
          c.block = b;
          c.vec_width = v;
          c.mode = m;
          grid.push_back(c);
        }
    (void)opts;
    return grid;
  }
  static ConfigVerdict validate(const Config &c, const Problem &p) {
    return validate_config(c, p);
  }
  static std::int64_t smem_bytes(const Config &, const Problem &) {
    return ElemCommon::smem_bytes();
  }
  static LaunchGeometry geometry(const Config &c, const Problem &p) {
    return launch_geometry(c, p);
  }
  static std::string symbol(const Config &c) { return kernel_symbol(c); }
  static bool generate(const Config &c, const Problem &p,
                       const std::string &path, std::string *err) {
    return generate_to_file(c, p, path, err);
  }
  static int block_size(const Config &c, const Problem &) { return c.block; }
  static int estimated_regs(const Config &, const Problem &) {
    return ElemCommon::estimated_regs();
  }
  static db::json::Value config_json(const Config &c) {
    return ElemCommon::config_json(c.block, c.vec_width, elem_mode_name(c.mode));
  }
  static std::vector<std::string> bench_bins() { return {"x.bin"}; }
  static std::vector<std::int32_t> scalar_args(
      const std::vector<std::int64_t> &s) {
    return ElemCommon::row_col_args(s);
  }
  static std::size_t out_bytes(const std::vector<std::int64_t> &s) {
    return ElemCommon::out_bytes(s);
  }
  static std::vector<std::int64_t> gate_shape(const DriverOptions &o) {
    return ElemCommon::gate_shape(o);
  }
  static roofline::Shape roofline_shape(const DriverOptions &o) {
    return ElemCommon::roofline_shape(o);
  }
};

// --- layernorm ---------------------------------------------------------------

struct LayerNormTraits {
  static constexpr const char *kFamily = "layernorm";
  static constexpr const char *kGoldenFamily = "layernorm";
  static constexpr bool kCausal = false;
  static constexpr roofline::KernelFamily kRooflineFamily =
      roofline::KernelFamily::kLayerNorm;
  static constexpr std::size_t kShapeArity = 2;

  using Config = LayerNormConfig;
  using Problem = ElemProblem;

  static Problem make_problem(const DriverOptions &o) {
    return ElemCommon::make_problem(o);
  }
  static Problem make_problem_for(const std::vector<std::int64_t> &shape,
                                  const std::string &arch) {
    return ElemCommon::make_problem_for(shape, arch);
  }
  static std::string shape_string(const std::vector<std::int64_t> &s) {
    return ElemCommon::shape_string(s);
  }
  static std::vector<Config> enumerate(const DriverOptions &opts) {
    std::vector<Config> grid;
    for (const int b : kElemBlockChoices)
      for (const int v : kElemVecWidthChoices)
        for (const ElemMode m : {ElemMode::kOnePass, ElemMode::kTwoPass}) {
          Config c;
          c.block = b;
          c.vec_width = v;
          c.mode = m;
          grid.push_back(c);
        }
    (void)opts;
    return grid;
  }
  static ConfigVerdict validate(const Config &c, const Problem &p) {
    return validate_config(c, p);
  }
  static std::int64_t smem_bytes(const Config &, const Problem &) {
    return ElemCommon::smem_bytes();
  }
  static LaunchGeometry geometry(const Config &c, const Problem &p) {
    return launch_geometry(c, p);
  }
  static std::string symbol(const Config &c) { return kernel_symbol(c); }
  static bool generate(const Config &c, const Problem &p,
                       const std::string &path, std::string *err) {
    return generate_to_file(c, p, path, err);
  }
  static int block_size(const Config &c, const Problem &) { return c.block; }
  static int estimated_regs(const Config &, const Problem &) {
    return ElemCommon::estimated_regs();
  }
  static db::json::Value config_json(const Config &c) {
    return ElemCommon::config_json(c.block, c.vec_width, elem_mode_name(c.mode));
  }
  static std::vector<std::string> bench_bins() {
    return {"x.bin", "gamma.bin", "beta.bin"};
  }
  static std::vector<std::int32_t> scalar_args(
      const std::vector<std::int64_t> &s) {
    return ElemCommon::row_col_args(s);
  }
  static std::size_t out_bytes(const std::vector<std::int64_t> &s) {
    return ElemCommon::out_bytes(s);
  }
  static std::vector<std::int64_t> gate_shape(const DriverOptions &o) {
    return ElemCommon::gate_shape(o);
  }
  static roofline::Shape roofline_shape(const DriverOptions &o) {
    return ElemCommon::roofline_shape(o);
  }
};

// --- reduction ---------------------------------------------------------------

struct ReductionTraits {
  static constexpr const char *kFamily = "reduction";
  static constexpr const char *kGoldenFamily = "reduction";
  static constexpr bool kCausal = false;
  static constexpr roofline::KernelFamily kRooflineFamily =
      roofline::KernelFamily::kReduction;
  static constexpr std::size_t kShapeArity = 2;

  using Config = ReductionConfig;
  using Problem = ElemProblem;

  static Problem make_problem(const DriverOptions &o) {
    return ElemCommon::make_problem(o);
  }
  static Problem make_problem_for(const std::vector<std::int64_t> &shape,
                                  const std::string &arch) {
    return ElemCommon::make_problem_for(shape, arch);
  }
  static std::string shape_string(const std::vector<std::int64_t> &s) {
    return ElemCommon::shape_string(s);
  }
  static std::vector<Config> enumerate(const DriverOptions &opts) {
    std::vector<Config> grid;
    for (const int b : kElemBlockChoices)
      for (const int v : kElemVecWidthChoices)
        for (const ElemMode m : {ElemMode::kStrided, ElemMode::kTree}) {
          Config c;
          c.block = b;
          c.vec_width = v;
          c.mode = m;
          grid.push_back(c);
        }
    (void)opts;
    return grid;
  }
  static ConfigVerdict validate(const Config &c, const Problem &p) {
    return validate_config(c, p);
  }
  static std::int64_t smem_bytes(const Config &, const Problem &) {
    return ElemCommon::smem_bytes();
  }
  static LaunchGeometry geometry(const Config &c, const Problem &p) {
    return launch_geometry(c, p);
  }
  static std::string symbol(const Config &c) { return kernel_symbol(c); }
  static bool generate(const Config &c, const Problem &p,
                       const std::string &path, std::string *err) {
    return generate_to_file(c, p, path, err);
  }
  static int block_size(const Config &c, const Problem &) { return c.block; }
  static int estimated_regs(const Config &, const Problem &) {
    return ElemCommon::estimated_regs();
  }
  static db::json::Value config_json(const Config &c) {
    return ElemCommon::config_json(c.block, c.vec_width, elem_mode_name(c.mode));
  }
  static std::vector<std::string> bench_bins() { return {"x.bin"}; }
  static std::vector<std::int32_t> scalar_args(
      const std::vector<std::int64_t> &s) {
    return ElemCommon::row_col_args(s);
  }
  static std::size_t out_bytes(const std::vector<std::int64_t> &s) {
    return ElemCommon::out_bytes(s);
  }
  static std::vector<std::int64_t> gate_shape(const DriverOptions &o) {
    return ElemCommon::gate_shape(o);
  }
  static roofline::Shape roofline_shape(const DriverOptions &o) {
    return ElemCommon::roofline_shape(o);
  }
};

}  // namespace

DriverResult tune_bias_gelu(const DriverOptions &opts, const DriverHooks *h) {
  return Tuner<BiasGeluTraits>::run(opts, h);
}
DriverResult tune_softmax(const DriverOptions &opts, const DriverHooks *h) {
  return Tuner<SoftmaxTraits>::run(opts, h);
}
DriverResult tune_layernorm(const DriverOptions &opts, const DriverHooks *h) {
  return Tuner<LayerNormTraits>::run(opts, h);
}
DriverResult tune_reduction(const DriverOptions &opts, const DriverHooks *h) {
  return Tuner<ReductionTraits>::run(opts, h);
}

}  // namespace driver_detail
}  // namespace kernscope::tune
