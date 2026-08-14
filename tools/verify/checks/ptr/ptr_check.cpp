// kernscope::verify::ptr — PointerDisciplineCheckPlugin (todo 34): check
// classes 5 (illegal pointer arithmetic) + 7 (uninitialized reads) + 8
// (invalid address-space casts).
//
// Registered via REGISTER_CHECK: the static registrar hands an instance to
// the global CheckRegistry — no shared file edits (todo 30 contract).
//
// SKELETON COMMIT: the plugin registers id "ptr" with check_class_ids
// {5, 7, 8} and returns an empty diagnostic vector. The class analyses land
// in follow-up commits (5 -> 8 -> 7), each with corpus fixtures + tests.

#include <kernscope/verify/check_plugin.h>

#include <string>
#include <vector>

namespace kernscope {
namespace verify {
namespace ptr {

class PointerDisciplineCheckPlugin : public CheckPlugin {
 public:
  PointerDisciplineCheckPlugin() {
    id = "ptr";
    name = "Pointer discipline: arithmetic bounds + address-space casts + "
           "uninitialized reads";
    check_class_ids = {CheckClass::kPtrArith, CheckClass::kUninitRead,
                       CheckClass::kAddrSpaceCast};
  }

  std::vector<Diagnostic> run(VerifierContext& ctx) const override {
    (void)ctx;
    return {};
  }
};

REGISTER_CHECK(PointerDisciplineCheckPlugin);

}  // namespace ptr
}  // namespace verify
}  // namespace kernscope
