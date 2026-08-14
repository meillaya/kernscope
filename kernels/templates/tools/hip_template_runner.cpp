//===- hip_template_runner.cpp - generic template-kernel live runner -------===//
//
// kernscope kernels/templates (todo 24). Loads a hipcc --genco hsaco through
// the kernscope::gpuruntime abstraction (libs/gpuruntime, todo 21) and
// launches ONE kernel with raw host buffers, proving the whole
// module/kernel/launch/memcpy path live on this machine (gfx1101).
//
// CLI contract (consumed by tests/run_hip_gate.py; documented in README.md):
//
//   hip_template_runner --hsaco <file.hsaco> --kernel <symbol>
//       --grid <W>[x<H>] --block <X>[x<Y>]
//       [--in <raw.bin>]... --out <out.bin> --out-size <bytes>
//       [--int <i32>]...
//
// Kernel parameter contract: the kernel signature is
//
//     (const T* input..., T* output, int32 scalar...)
//
// i.e. device pointers for every --in in order, then the --out device
// pointer, then the --int scalars in order, all passed BY VALUE (uintptr for
// pointers) per the driver-API convention proven by libs/gpuruntime (todo 21;
// gpuruntime-hip-test AddKernelLaunchMemcpyRoundtrip). Input/output buffers
// are opaque bytes (the templates are bf16, 2 bytes/element).
//
// Exit codes: 0 ok; non-zero with the typed error + backend message on any
// failure (no silent skips - a gate must fail loudly).
//
//===----------------------------------------------------------------------===//

#include "kernscope/gpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string hsaco;
  std::string kernel;
  std::uint32_t grid_x = 1;
  std::uint32_t grid_y = 1;
  std::uint32_t block_x = 1;
  std::uint32_t block_y = 1;
  std::vector<std::string> inputs;
  std::string out;
  std::size_t out_size = 0;
  std::vector<std::int32_t> scalars;
};

void usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s --hsaco <f> --kernel <sym> --grid <W[xH]> "
               "--block <X[xY]> [--in <bin>]... --out <bin> --out-size <bytes> "
               "[--int <i32>]...\n",
               argv0);
}

// Parse a "W" or "WxH" dimension string into two positive uint32s.
bool parse_dims(const std::string &s, std::uint32_t *a, std::uint32_t *b) {
  const std::size_t x = s.find('x');
  const std::string sa = x == std::string::npos ? s : s.substr(0, x);
  const std::string sb = x == std::string::npos ? "1" : s.substr(x + 1);
  if (sa.empty() || sb.empty())
    return false;
  try {
    const long va = std::stol(sa);
    const long vb = std::stol(sb);
    if (va <= 0 || vb <= 0 || va > 0xFFFFFFFFL || vb > 0xFFFFFFFFL)
      return false;
    *a = static_cast<std::uint32_t>(va);
    *b = static_cast<std::uint32_t>(vb);
  } catch (...) {
    return false;
  }
  return true;
}

bool read_file(const std::string &path, std::vector<char> *out) {
  std::ifstream fh(path, std::ios::binary);
  if (!fh)
    return false;
  fh.seekg(0, std::ios::end);
  const std::streamsize size = fh.tellg();
  fh.seekg(0, std::ios::beg);
  if (size < 0)
    return false;
  out->resize(static_cast<std::size_t>(size));
  if (size > 0)
    fh.read(out->data(), size);
  return static_cast<bool>(fh) || fh.eof();
}

bool write_file(const std::string &path, const std::vector<char> &data) {
  std::ofstream fh(path, std::ios::binary | std::ios::trunc);
  if (!fh)
    return false;
  fh.write(data.data(), static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(fh);
}

int fail(const std::string &what) {
  std::fprintf(stderr, "hip_template_runner: %s\n", what.c_str());
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "hip_template_runner: %s requires an argument\n", name);
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--hsaco")
      opt.hsaco = need("--hsaco");
    else if (a == "--kernel")
      opt.kernel = need("--kernel");
    else if (a == "--grid") {
      if (!parse_dims(need("--grid"), &opt.grid_x, &opt.grid_y)) {
        std::fprintf(stderr, "hip_template_runner: bad --grid value\n");
        return 1;
      }
    } else if (a == "--block") {
      if (!parse_dims(need("--block"), &opt.block_x, &opt.block_y)) {
        std::fprintf(stderr, "hip_template_runner: bad --block value\n");
        return 1;
      }
    } else if (a == "--in")
      opt.inputs.push_back(need("--in"));
    else if (a == "--out")
      opt.out = need("--out");
    else if (a == "--out-size") {
      try {
        const long v = std::stol(need("--out-size"));
        if (v < 0)
          throw std::out_of_range("negative");
        opt.out_size = static_cast<std::size_t>(v);
      } catch (...) {
        std::fprintf(stderr, "hip_template_runner: bad --out-size value\n");
        return 1;
      }
    } else if (a == "--int") {
      try {
        opt.scalars.push_back(static_cast<std::int32_t>(std::stol(need("--int"))));
      } catch (...) {
        std::fprintf(stderr, "hip_template_runner: bad --int value\n");
        return 1;
      }
    } else {
      std::fprintf(stderr, "hip_template_runner: unknown option '%s'\n", a.c_str());
      usage(argv[0]);
      return 1;
    }
  }
  if (opt.hsaco.empty() || opt.kernel.empty() || opt.out.empty() ||
      opt.out_size == 0) {
    usage(argv[0]);
    return 1;
  }

  auto rt = kernscope::create_runtime(kernscope::Backend::kHip);
  if (!rt.ok())
    return fail("create_runtime(hip): " + rt.message);
  std::printf("runner: backend=%s\n", std::string(rt.value->backend_name()).c_str());

  auto mod = kernscope::Module::load(*rt.value, opt.hsaco);
  if (!mod.ok())
    return fail("Module::load(" + opt.hsaco + "): " + mod.message);
  auto kern = kernscope::Kernel::create(*rt.value, *mod.value, opt.kernel);
  if (!kern.ok())
    return fail("Kernel::create(" + opt.kernel + "): " + kern.message);

  // Upload every input buffer.
  std::vector<std::vector<char>> blobs;
  std::vector<kernscope::DevicePtr> in_ptrs;
  for (const std::string &path : opt.inputs) {
    blobs.emplace_back();
    if (!read_file(path, &blobs.back()))
      return fail("cannot read input " + path);
    auto d = rt.value->device_malloc(blobs.back().size());
    if (!d.ok())
      return fail("device_malloc(" + path + "): " + d.message);
    if (!rt.value->memcpy_htod(d.value, blobs.back().data(), blobs.back().size()).ok())
      return fail("memcpy_htod(" + path + ") failed");
    in_ptrs.push_back(d.value);
    std::printf("runner: input %s: %zu bytes -> device\n", path.c_str(),
                blobs.back().size());
  }

  auto out_d = rt.value->device_malloc(opt.out_size);
  if (!out_d.ok())
    return fail("device_malloc(out): " + out_d.message);

  // Driver-API convention: device pointers pass BY VALUE (uintptr).
  std::vector<std::uint64_t> ptrs;
  for (const kernscope::DevicePtr &p : in_ptrs)
    ptrs.push_back(p.value);
  ptrs.push_back(out_d.value.value);

  std::vector<kernscope::KernelArg> args;
  for (const std::uint64_t &p : ptrs)
    args.push_back(kernscope::KernelArg{&p, sizeof(p)});
  for (const std::int32_t &s : opt.scalars)
    args.push_back(kernscope::KernelArg{&s, sizeof(s)});

  kernscope::LaunchConfig cfg;
  cfg.grid_x = opt.grid_x;
  cfg.grid_y = opt.grid_y;
  cfg.block_x = opt.block_x;
  cfg.block_y = opt.block_y;
  std::printf("runner: launch %s grid=%ux%u block=%ux%u args=%zu\n",
              opt.kernel.c_str(), cfg.grid_x, cfg.grid_y, cfg.block_x,
              cfg.block_y, args.size());
  auto launch = kern.value->launch(cfg, args);
  if (!launch.ok())
    return fail("launch(" + opt.kernel + "): " + launch.message);
  if (!rt.value->synchronize().ok())
    return fail("synchronize failed");

  std::vector<char> out(opt.out_size);
  if (!rt.value->memcpy_dtoh(out.data(), out_d.value, opt.out_size).ok())
    return fail("memcpy_dtoh(out) failed");
  if (!write_file(opt.out, out))
    return fail("cannot write output " + opt.out);
  std::printf("runner: wrote %zu bytes -> %s\n", opt.out.size(), opt.out.c_str());

  for (const kernscope::DevicePtr &p : in_ptrs)
    (void)rt.value->device_free(p);
  (void)rt.value->device_free(out_d.value);
  return 0;
}
