#include <kernscope/verify/budget.h>

#include <string>

namespace kernscope {
namespace verify {

void Budget::reset(std::string_view function) {
  std::string key(function);
  consumed_[key] = 0;
  latched_[key] = false;
}

bool Budget::acquire(std::string_view function, uint64_t n) {
  std::string key(function);
  if (latched_[key]) {
    return false;
  }
  uint64_t& used = consumed_[key];
  if (used + n > max_states_per_function_) {
    latched_[key] = true;
    return false;
  }
  used += n;
  return true;
}

bool Budget::exceeded(std::string_view function) const {
  auto it = latched_.find(std::string(function));
  return it != latched_.end() && it->second;
}

uint64_t Budget::consumed(std::string_view function) const {
  auto it = consumed_.find(std::string(function));
  return it == consumed_.end() ? 0 : it->second;
}

}  // namespace verify
}  // namespace kernscope
