//===- equiv_cuda_side.cpp - equivalence test: CUDA-side bundle producer -------*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// The CUDA side of the todo-44 dual-mechanism equivalence test
// (ctest -R capture_equiv -L gpu-amd): produces the bundle the CUPTI
// mechanism would record for the todo-24 elementwise_add program, driven by
// the synthetic CUPTI-callback harness (tests/synthetic_cupti.h) - the
// "CUDA-shim-on-mock" side. The HIP side runs the SAME program live on
// gfx1101 under libkernscope_capture_hip.so (todo 45); capture_equiv_check.py
// compares the two bundles field-for-field per docs/capture-mapping.md.
//
// The replayed program mirrors tools/hip_template_runner.cpp exactly:
//   hipMalloc a (2048) -> H2D a -> hipMalloc b (2048) -> H2D b ->
//   hipMalloc out (2048) -> launch (grid 4, block 256,
//   args [a, b, out, int32 n]) -> synchronize -> D2H out -> free a, b, out
//
// The CUPTI events stand in for what real CUPTI delivers (no NVIDIA runtime
// here; rental-gated on real hardware):
//   * CUpti_ActivityMemory4 records -> allocation sizes (M9 in the mapping);
//   * RUNTIME cudaMemcpy ENTER/EXIT   -> copy records + input-buffer blobs;
//   * RUNTIME cudaLaunchKernel ENTER  -> launch record (M1-M8);
//   * DRIVER cuModuleGetFunction/cuFuncGetParamInfo EXIT -> layout facts;
//   * RESOURCE module-loaded          -> cubin blob (M12/M13).
//
// Device addresses are SYNTHETIC (0x1000/0x2000/0x3000): the comparison
// VA-normalizes them to allocation ordinals, so the values themselves never
// cross the equivalence boundary (docs/capture-mapping.md §4.3).
//
//===----------------------------------------------------------------------===//

#include "kernscope/cupti_capture.h"
#include "synthetic_cupti.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using kernscope::LaunchVia;
namespace syn = kernscope::cupti::synthetic;

constexpr CUpti_CallbackDomain kRuntime = CUPTI_CB_DOMAIN_RUNTIME_API;
constexpr CUpti_CallbackDomain kDriver = CUPTI_CB_DOMAIN_DRIVER_API;
constexpr CUpti_CallbackDomain kResource = CUPTI_CB_DOMAIN_RESOURCE;

struct Options {
  std::string cubin;
  std::string kernel_name;
  std::string bundle;
  std::string blob_dir;
  std::vector<std::string> inputs;
  std::size_t out_size = 0;
  std::int32_t n = 0;
  std::uint32_t grid_x = 1, block_x = 1;
};

int Fail(const std::string &what) {
  std::fprintf(stderr, "equiv_cuda_side: %s\n", what.c_str());
  return 1;
}

std::vector<std::uint8_t> ReadFile(const std::string &path) {
  std::ifstream fh(path, std::ios::binary);
  if (!fh) {
    return {};
  }
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(fh)),
                                   std::istreambuf_iterator<char>());
}

/// One CUpti_ActivityMemory4 delivery (device allocation / release).
void SendMemoryRecord(std::uint64_t address, std::uint64_t bytes, bool alloc) {
  CUpti_ActivityMemory4 rec{};
  rec.kind = CUPTI_ACTIVITY_KIND_MEMORY;
  rec.memoryOperationType = alloc
      ? CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION
      : CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE;
  rec.memoryKind = CUPTI_ACTIVITY_MEMORY_KIND_DEVICE;
  rec.address = address;
  rec.bytes = bytes;
  kernscope::cupti::handle_activity_record(
      reinterpret_cast<const CUpti_Activity *>(&rec));
}

/// RUNTIME cudaMemcpy ENTER+EXIT (kind 1 = HostToDevice, 2 = DeviceToHost).
void SendMemcpy(std::uint64_t dst, const void *src, std::size_t bytes,
                int kind) {
  cudaMemcpy_v3020_params p;
  p.dst = reinterpret_cast<void *>(dst);
  p.src = src;
  p.count = bytes;
  p.kind = static_cast<cudaMemcpyKind>(kind);
  const CUpti_CallbackId cbid =
      static_cast<CUpti_CallbackId>(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020);
  auto cb = syn::CudaMemcpy(&p, cudaSuccess);
  cb.At(CUPTI_API_ENTER)->Send(kRuntime, cbid);
  cb.At(CUPTI_API_EXIT)->Send(kRuntime, cbid);
}

} // namespace

int main(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "equiv_cuda_side: %s requires an argument\n",
                     name);
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--cubin")
      opt.cubin = need("--cubin");
    else if (a == "--kernel-name")
      opt.kernel_name = need("--kernel-name");
    else if (a == "--bundle")
      opt.bundle = need("--bundle");
    else if (a == "--blobdir")
      opt.blob_dir = need("--blobdir");
    else if (a == "--in")
      opt.inputs.push_back(need("--in"));
    else if (a == "--out-size")
      opt.out_size = static_cast<std::size_t>(std::stoul(need("--out-size")));
    else if (a == "--n")
      opt.n = static_cast<std::int32_t>(std::stol(need("--n")));
    else if (a == "--grid")
      opt.grid_x = static_cast<std::uint32_t>(std::stoul(need("--grid")));
    else if (a == "--block")
      opt.block_x = static_cast<std::uint32_t>(std::stoul(need("--block")));
    else {
      std::fprintf(stderr, "equiv_cuda_side: unknown option %s\n", a.c_str());
      return 1;
    }
  }
  if (opt.cubin.empty() || opt.kernel_name.empty() || opt.bundle.empty() ||
      opt.blob_dir.empty() || opt.inputs.size() != 2 || opt.out_size == 0) {
    std::fprintf(stderr,
                 "usage: equiv_cuda_side --cubin <f> --kernel-name <n> "
                 "--bundle <out> --blobdir <d> --in <a> --in <b> "
                 "--out-size <bytes> --n <i32> --grid <g> --block <b>\n");
    return 1;
  }
  if (opt.inputs[0] == opt.inputs[1]) {
    return Fail("inputs must be two distinct files");
  }

  // Blob output (cubin-<id>.cubin + input-buffer-<n>.bin) goes to the
  // per-side blob dir (the KERNSCOPE_CUPTI_BLOB_DIR contract).
  setenv(kernscope::cupti::kBlobDirEnv, opt.blob_dir.c_str(), 1);

  kernscope::CaptureSession session;
  kernscope::cupti::arm_session(&session);

  // 0. Module load FIRST (the app loads the kernel image before anything
  //    else): the real cubin compiled from the todo-24 source for sm_80.
  const std::vector<std::uint8_t> cubin = ReadFile(opt.cubin);
  if (cubin.empty()) {
    return Fail("cannot read cubin " + opt.cubin);
  }
  auto mod_cb = syn::ModuleLoaded(cubin);
  mod_cb.Send(kResource, static_cast<CUpti_CallbackId>(
                             CUPTI_CBID_RESOURCE_MODULE_LOADED));

  // 1. Driver-API layout facts: cuModuleGetFunction maps the CUfunction to
  //    the kernel name; cuFuncGetParamInfo reports the 4-param layout
  //    (ptr a, ptr b, ptr out, int32 n - the kernel image's parameter order).
  constexpr std::uint64_t kCufunc = 0xABCD;
  auto modfn = syn::CuModuleGetFunction(kCufunc, opt.kernel_name.c_str());
  modfn.At(CUPTI_API_EXIT)->Send(kDriver, static_cast<CUpti_CallbackId>(
                          CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction));
  const std::size_t offsets[] = {0, 8, 16, 24};
  const std::size_t sizes[] = {8, 8, 8, 4};
  for (int i = 0; i < 4; ++i) {
    auto cb = syn::CuFuncGetParamInfo(kCufunc, i, offsets[i], sizes[i]);
    cb.At(CUPTI_API_EXIT)->Send(kDriver, static_cast<CUpti_CallbackId>(
                         CUPTI_DRIVER_TRACE_CBID_cuFuncGetParamInfo));
  }

  // 2. The program: synthetic device VAs (0x1000/0x2000/0x3000) in the same
  //    allocation order as the HIP runner (a, b, out) - VA normalization
  //    (§4.3) maps both sides to allocation ordinals.
  const std::vector<std::uint8_t> a = ReadFile(opt.inputs[0]);
  const std::vector<std::uint8_t> b = ReadFile(opt.inputs[1]);
  if (a.size() != opt.out_size || b.size() != opt.out_size) {
    return Fail("input sizes must equal --out-size");
  }
  const std::uint64_t d_a = 0x1000, d_b = 0x2000, d_out = 0x3000;
  std::vector<std::uint8_t> host_out(opt.out_size, 0);

  SendMemoryRecord(d_a, opt.out_size, /*alloc=*/true);
  SendMemcpy(d_a, a.data(), a.size(), 1); // H2D a -> input-buffer-1.bin
  SendMemoryRecord(d_b, opt.out_size, /*alloc=*/true);
  SendMemcpy(d_b, b.data(), b.size(), 1); // H2D b -> input-buffer-2.bin
  SendMemoryRecord(d_out, opt.out_size, /*alloc=*/true);

  // 3. Launch (cudaLaunchKernel contract: args is an array of pointers to
  //    the parameter values; symbolName is the kernel name).
  std::int32_t n = opt.n;
  void *args[] = {const_cast<std::uint64_t *>(&d_a),
                  const_cast<std::uint64_t *>(&d_b),
                  const_cast<std::uint64_t *>(&d_out), &n};
  cudaLaunchKernel_v7000_params lp;
  lp.func = reinterpret_cast<const void *>(0xFEED);
  lp.gridDim = dim3(opt.grid_x, 1, 1);
  lp.blockDim = dim3(opt.block_x, 1, 1);
  lp.args = args;
  lp.sharedMem = 0;
  lp.stream = nullptr;
  auto lc = syn::CudaLaunchKernel(&lp);
  lc.At(CUPTI_API_ENTER, opt.kernel_name.c_str())
      ->Send(kRuntime, static_cast<CUpti_CallbackId>(
                           CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

  // 4. Synchronize + D2H copy of the output.
  cudaStreamSynchronize_v3020_params sp;
  sp.stream = nullptr;
  auto scb = syn::CudaStreamSynchronize(&sp, cudaSuccess);
  scb.Send(kRuntime, static_cast<CUpti_CallbackId>(
                         CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020));
  SendMemcpy(reinterpret_cast<std::uint64_t>(host_out.data()), 
             reinterpret_cast<const void *>(d_out), opt.out_size, 2);

  // 5. Frees (the runner frees a, b, out at exit).
  cudaFree_v3020_params fp;
  cudaError_t rc = cudaSuccess;
  for (std::uint64_t ptr : {d_a, d_b, d_out}) {
    fp.devPtr = reinterpret_cast<void *>(ptr);
    auto cb = syn::CudaFree(&fp, rc);
    cb.At(CUPTI_API_EXIT)->Send(kRuntime, static_cast<CUpti_CallbackId>(
                             CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020));
  }

  // 6. Bundle: the SAME v0 BundleWriter the HIP tool's FlushBundle uses
  //    (libs/capture-core, never modified).
  std::ofstream out(opt.bundle, std::ios::trunc);
  if (!out) {
    return Fail("cannot write bundle " + opt.bundle);
  }
  kernscope::BundleWriter::create_v0()->write(session, out);
  std::printf("equiv_cuda_side: bundle written to %s (launches=%zu "
              "allocs=%zu copies=%zu)\n",
              opt.bundle.c_str(), session.launches().size(),
              session.allocs().size(), session.copies().size());
  return 0;
}
