//===- driver.cpp - tuning driver dispatch + keys --------------------------===//
//
// kernscope tools/tune todo 28. The public entry point: family dispatch,
// stable status names, and the Triton-style key helpers (arch:family:
// shape_hash — brief §1: re-tune when the keyed args change). The pipeline
// itself is family-generic (driver_pipeline_impl.h), instantiated per family
// in the adapter TUs. toolchain_paths() (todo 29) exposes the configure-time
// toolchain resolution for the CLI's --emit-repro.
//
//===----------------------------------------------------------------------===//

#include "kernscope/tune/driver.h"

#include "driver_families.h"
#include "driver_util.h"

#include <sstream>
#include <string>
#include <vector>

// Toolchain paths injected by tools/tune/CMakeLists.txt (absolute at
// configure time; empty when the piece was not found).
#ifndef KERNSCOPE_TUNE_NVCC
#define KERNSCOPE_TUNE_NVCC ""
#endif
#ifndef KERNSCOPE_TUNE_HIPCC
#define KERNSCOPE_TUNE_HIPCC ""
#endif
#ifndef KERNSCOPE_TUNE_HIP_INCLUDE
#define KERNSCOPE_TUNE_HIP_INCLUDE ""
#endif
#ifndef KERNSCOPE_TUNE_KERNEL_TRAITS_INCLUDE
#define KERNSCOPE_TUNE_KERNEL_TRAITS_INCLUDE ""
#endif
#ifndef KERNSCOPE_TUNE_REPO_ROOT
#define KERNSCOPE_TUNE_REPO_ROOT ""
#endif
#ifndef KERNSCOPE_TUNE_RUN_HIP_GATE
#define KERNSCOPE_TUNE_RUN_HIP_GATE ""
#endif
#ifndef KERNSCOPE_TUNE_PYTHON
#define KERNSCOPE_TUNE_PYTHON "python3"
#endif
#ifndef KERNSCOPE_TUNE_RUNNER
#define KERNSCOPE_TUNE_RUNNER ""
#endif

namespace kernscope::tune {

const char *driver_status_name(DriverStatus status) noexcept {
  switch (status) {
  case DriverStatus::kOk:
    return "ok";
  case DriverStatus::kNoViableConfig:
    return "no_viable_config";
  case DriverStatus::kNoLiveGpu:
    return "no_live_gpu";
  case DriverStatus::kUsageError:
    return "usage_error";
  case DriverStatus::kInternalError:
    return "internal_error";
  }
  return "unknown";
}

std::string make_shape_hash(const std::string &kernel_family,
                            const std::vector<std::int64_t> &shape) {
  std::ostringstream key;
  key << kernel_family << ":";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      key << ",";
    key << shape[i];
  }
  return "fnv1a64:" + driver_detail::to_hex(driver_detail::fnv1a64(key.str()));
}

std::string make_cache_key(const std::string &arch,
                           const std::string &kernel_family,
                           const std::string &shape_hash) {
  return arch + ":" + kernel_family + ":" + shape_hash;
}

ToolchainPaths toolchain_paths() noexcept {
  ToolchainPaths p;
  p.nvcc = KERNSCOPE_TUNE_NVCC;
  p.hipcc = KERNSCOPE_TUNE_HIPCC;
  p.hip_include = KERNSCOPE_TUNE_HIP_INCLUDE;
  p.traits_include = KERNSCOPE_TUNE_KERNEL_TRAITS_INCLUDE;
  p.repo_root = KERNSCOPE_TUNE_REPO_ROOT;
  p.run_hip_gate = KERNSCOPE_TUNE_RUN_HIP_GATE;
  p.python = KERNSCOPE_TUNE_PYTHON;
  p.runner = KERNSCOPE_TUNE_RUNNER;
  return p;
}

DriverResult tune(const DriverOptions &opts, const DriverHooks *hooks) {
  using driver_detail::tune_bias_gelu;
  using driver_detail::tune_flash_attention;
  using driver_detail::tune_layernorm;
  using driver_detail::tune_matmul;
  using driver_detail::tune_reduction;
  using driver_detail::tune_softmax;

  if (opts.kernel_family == "matmul")
    return tune_matmul(opts, hooks);
  if (opts.kernel_family == "bias_gelu")
    return tune_bias_gelu(opts, hooks);
  if (opts.kernel_family == "softmax")
    return tune_softmax(opts, hooks);
  if (opts.kernel_family == "layernorm")
    return tune_layernorm(opts, hooks);
  if (opts.kernel_family == "reduction")
    return tune_reduction(opts, hooks);
  if (opts.kernel_family == "flash_attention")
    return tune_flash_attention(opts, hooks);

  DriverResult out;
  out.status = DriverStatus::kUsageError;
  out.message =
      "unknown kernel family '" + opts.kernel_family +
      "' (matmul | bias_gelu | softmax | layernorm | reduction | "
      "flash_attention)";
  return out;
}

} // namespace kernscope::tune
