#include <kernscope/verify/check_plugin.h>

#include <memory>
#include <vector>

namespace kernscope {
namespace verify {

CheckRegistry& CheckRegistry::instance() {
  static CheckRegistry registry;
  return registry;
}

void CheckRegistry::registerPlugin(std::unique_ptr<CheckPlugin> plugin) {
  plugins_.push_back(std::move(plugin));
}

const std::vector<std::unique_ptr<CheckPlugin>>& CheckRegistry::plugins() const {
  return plugins_;
}

std::vector<const CheckPlugin*> CheckRegistry::pluginsForClass(int class_id) const {
  std::vector<const CheckPlugin*> out;
  for (const auto& p : plugins_) {
    for (int id : p->check_class_ids) {
      if (id == class_id) {
        out.push_back(p.get());
        break;
      }
    }
  }
  return out;
}

std::vector<CheckRegistry::Entry> CheckRegistry::enumerate() const {
  std::vector<Entry> out;
  out.reserve(plugins_.size());
  for (const auto& p : plugins_) {
    out.push_back(Entry{p->id, p->name, p->check_class_ids});
  }
  return out;
}

CheckPluginRegistrar::CheckPluginRegistrar(std::unique_ptr<CheckPlugin> plugin) {
  CheckRegistry::instance().registerPlugin(std::move(plugin));
}

}  // namespace verify
}  // namespace kernscope
