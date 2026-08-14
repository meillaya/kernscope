//===- driver_families.h - per-family adapter entry points -----------------===//
//
// kernscope tools/tune todo 28. The six per-family entry points the public
// tune() dispatch calls. Each is implemented in its own TU so that no TU
// mixes generator headers (each generator declares its own
// ConfigVerdict/LaunchGeometry in kernscope::tune — ODR-hostile to mix).
// Internal to kernscope_tune_driver (never installed).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_TUNE_DRIVER_FAMILIES_H
#define KERNSCOPE_TUNE_DRIVER_FAMILIES_H

#include "kernscope/tune/driver.h"

namespace kernscope::tune {
namespace driver_detail {

DriverResult tune_matmul(const DriverOptions &opts, const DriverHooks *hooks);
DriverResult tune_bias_gelu(const DriverOptions &opts, const DriverHooks *hooks);
DriverResult tune_softmax(const DriverOptions &opts, const DriverHooks *hooks);
DriverResult tune_layernorm(const DriverOptions &opts, const DriverHooks *hooks);
DriverResult tune_reduction(const DriverOptions &opts, const DriverHooks *hooks);
DriverResult tune_flash_attention(const DriverOptions &opts,
                                  const DriverHooks *hooks);

}  // namespace driver_detail
}  // namespace kernscope::tune

#endif // KERNSCOPE_TUNE_DRIVER_FAMILIES_H
