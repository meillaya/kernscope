// kernscope::pass — pass-module registry implementation (todo 36).
//
// Defines the singleton registry + registrar that the REGISTER_PASS_CALLBACK
// macro (register_pass.h) and the plugin entry (src/plugin.cpp) share. The
// registry itself is a plain std::vector<std::function<void(llvm::
// PassBuilder&)>>; the plugin entry invokes every callback with the
// PassBuilder it is handed at dlopen time.

#include <utility>

#include <kernscope/pass/register_pass.h>

namespace kernscope {
namespace pass {

PassRegistry& PassRegistry::instance() {
  static PassRegistry registry;
  return registry;
}

void PassRegistry::registerCallback(PassRegistrationCallback cb) {
  callbacks_.push_back(std::move(cb));
}

const std::vector<PassRegistrationCallback>& PassRegistry::callbacks() const {
  return callbacks_;
}

PassRegistrar::PassRegistrar(PassRegistrationCallback cb) {
  PassRegistry::instance().registerCallback(std::move(cb));
}

}  // namespace pass
}  // namespace kernscope
