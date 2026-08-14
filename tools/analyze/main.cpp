// kernscope-analyze: static GPU-kernel analyzer (stub).
//
// Wave-3 fills in the real analyses. This stub proves the target graph and
// the ctest LABELS convention under -Wall -Wextra -Werror.
#include <cstdio>
#include <string_view>

#include "kernscope/version.h"

namespace {

void printVersion() {
  std::printf("kernscope-analyze %s (kernscope %s)\n", KERNSCOPE_VERSION_STRING,
              KERNSCOPE_VERSION_STRING);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--version") {
    printVersion();
    return 0;
  }
  std::printf("kernscope-analyze: static GPU-kernel analyzer (stub). "
              "Run with --version.\n");
  return 0;
}
