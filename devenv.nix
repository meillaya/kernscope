{ pkgs, lib, config, inputs, ... }:

# kernscope dev shell — LLVM/MLIR 21 + CUDA 12.6 redistributables + ROCm 7.2.3.
# Pattern: /home/mei/projects/polykernel/devenv.nix (proven on this machine),
# with the review-mandated single-root CUDA_HOME deviation (see below).
#
# nixpkgs config note (allowUnfree / cudaSupport / rocmSupport / cudaCapabilities)
# is applied via the devenv.yaml `nixpkgs.*` options (devenv >= 2.2.1 here),
# NOT here: the devenv module system has NO `nixpkgs.config` option (polykernel
# verified on v1.11.2/v2.1.2 — setting `nixpkgs.config.*` in a module errors
# with "The option `nixpkgs' does not exist").
#
# FALLBACK BRANCH (only if the yaml nixpkgs.* route errors on this devenv):
# drop the `nixpkgs:` block from devenv.yaml and add a polykernel-pattern
# flake.nix importing nixpkgs with
#   config = { allowUnfree = true; cudaSupport = true; rocmSupport = true;
#              cudaCapabilities = ["8.0" "9.0"]; };
# (/home/mei/projects/polykernel/flake.nix:26-36). Toggling allowUnfree=false
# there makes cudaPackages fail to evaluate (the CUDA EULA gate), which is the
# QA negative test.

let
  # LLVM/MLIR 21 toolchain (nixpkgs-unstable treewide default).
  llvmPkgs = pkgs.llvmPackages_21;

  # CUDA 12.6 redistributables (compile/analyze-only path locally; sm_80/sm_90).
  cudaPackages = pkgs.cudaPackages_12_6;

  # ROCm/HIP. clr provides hipcc + the HIP runtime and auto-sets
  # HIP_PATH/ROCM_PATH/HIP_CLANG_PATH/DEVICE_LIB_PATH/HSA_PATH; gpuTargets
  # include gfx1101 (RX 7800 XT, officially supported since ROCm 7.0).
  # No manual ROCM_PATH needed — see docs/hip_backend.md recipe (polykernel).
  rocmPackages = pkgs.rocmPackages;

  # Python harness for golden/DB tooling.
  python = pkgs.python3.withPackages (ps: with ps; [
    numpy
    ml-dtypes
    pytest
  ]);
in
{
  packages = [
    # --- LLVM/MLIR 21 (out-of-tree dialect build) ---
    llvmPkgs.mlir      # MLIR libs + headers (.dev: lib/cmake/mlir)
    llvmPkgs.tblgen    # mlir-tblgen
    llvmPkgs.llvm      # FileCheck, llvm-objdump, llvm-readobj
    llvmPkgs.clang     # CC / CXX

    # --- Build tools ---
    pkgs.lit           # LLVM Integrated Tester (lit)
    pkgs.cmake
    pkgs.ninja

    # --- CUDA redistributables (preferred over monolithic cudatoolkit) ---
    cudaPackages.cuda_nvcc        # nvcc + ptxas
    cudaPackages.cuda_cudart      # runtime + headers (CUDA_HOME root)
    cudaPackages.cuda_nvrtc       # runtime compilation
    cudaPackages.libcublas
    cudaPackages.cuda_cuobjdump
    cudaPackages.cuda_cupti       # CUPTI profiler headers/libs (CUPTI_HOME)
    cudaPackages.cuda_nvdisasm    # cubin disassembler

    # --- ROCm / HIP (first-class local run on gfx1101) ---
    rocmPackages.clr              # hipcc + HIP runtime
    rocmPackages.rocminfo
    rocmPackages.rocm-smi
    rocmPackages.rocprofiler-sdk  # ROCm profiler SDK (NOT deprecated roctracer)

    # --- Unit testing (REQUIRED: gtest unit tests won't configure without it) ---
    pkgs.gtest

    # --- Python golden harness ---
    python
  ];

  # Use llvmPackages_21.clang for CC/CXX (NOT clang_18 - avoid MLIR-21 header
  # skew). CC = clang, CXX = clang++ (the C++ driver of the same package, so
  # CMake C++ targets link the C++ runtime correctly).
  env.CC = lib.getExe llvmPkgs.clang;
  env.CXX = "${llvmPkgs.clang}/bin/clang++";

  # CRITICAL DEVIATION FROM POLYKERNEL (review round-1 fix, Oracle HIGH#2 /
  # Momus HIGH#1): CUDA_HOME is a SINGLE toolkit root — NOT a colon-separated
  # multi-path list. polykernel defines CUDA_HOME at devenv.nix:82-87 via
  # lib.makeSearchPathOutput over four packages; we deliberately deviate
  # because downstream consumers (clang's --cuda-path, CMake's FindCUDA, nvcc
  # driver mode) require one flat root.
  #
  # In this nixpkgs rev cudaPackages_12_6.cuda_cudart has a SINGLE `out`
  # output (verified 2026-08-14: cuda_cudart-12.6.77, no `.dev` output) whose
  # tree is include/cuda_runtime.h + include/cuda.h + lib/libcudart.so* +
  # nvvm/ — exactly one toolkit root. If a future rev splits outputs and the
  # root lacks any needed header, replace it with
  #   pkgs.symlinkJoin { name = "cuda-home"; paths = [ cuda_cudart.out ... ]; }
  # as the ONE root — never a colon-separated CUDA_HOME.
  env.CUDA_HOME = "${cudaPackages.cuda_cudart}";

  # CUPTI profiler headers/libs (single root, same discipline). cuda_cupti
  # splits outputs in nixpkgs: `include` = headers (cupti.h,
  # cupti_activity.h), `lib` = lib/libcupti.so*, `dev` = only .pc/.cmake.
  # Union the two needed outputs into ONE root via symlinkJoin.
  env.CUPTI_HOME = "${pkgs.symlinkJoin {
    name = "cuda-cupti-home";
    paths = [ cudaPackages.cuda_cupti.include cudaPackages.cuda_cupti.lib ];
  }}";

  # clang CUDA detection fallback: this machine has no /opt/cuda, so point
  # clang's --cuda-path lookup at the same single root.
  env.CUDA_PATH = "$CUDA_HOME";

  # rocprofiler-sdk output pointers (single paths, no colons) for the
  # tests/env/rocprof_stub.c compile gate: headers live in the `dev` output
  # (include/rocprofiler-sdk/), librocprofiler-sdk.so in the `out` output.
  env.ROCPROF_SDK_DEV = "${rocmPackages.rocprofiler-sdk.dev}";
  env.ROCPROF_SDK_OUT = "${rocmPackages.rocprofiler-sdk}";

  # Runtime library path: CUDA libs + HIP runtime + the system graphics driver.
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    cudaPackages.cuda_cudart
    cudaPackages.libcublas
    cudaPackages.cuda_nvrtc
    rocmPackages.clr
    "/run/opengl-driver"   # system Mesa/vulkan driver
  ];

  enterShell = ''
    # nixpkgs stdenv's setup hook exports CC/CXX=gcc/g++; re-export the LLVM-21
    # clang here (enterShell runs last) so the compiler the CMake build picks up
    # is llvmPackages_21.clang, matching MLIR-21 headers.
    export CC="${lib.getExe llvmPkgs.clang}"
    export CXX="${llvmPkgs.clang}/bin/clang++"
    # ROCm runtime gate (polykernel docs/hip_backend.md recipe):
    # - clr auto-sets HIP_PATH/ROCM_PATH/HIP_CLANG_PATH/DEVICE_LIB_PATH/HSA_PATH
    #   (no manual ROCM_PATH needed);
    # - HSA_OVERRIDE_GFX_VERSION=11.0.1 is a FALLBACK ONLY: some HSA runtimes
    #   do not recognize gfx1101 by name; gfx1101 is ISA-compatible with
    #   gfx1100 (same RDNA3 ISA + register file), so 11.0.1/11.0.0 emulation is
    #   safe. Only export it if rocminfo fails to enumerate gfx1101 natively.
    # - the user must be in the video AND render groups to access /dev/kfd and
    #   /dev/dri/renderD* (check: `groups`; remedy: sudo usermod -aG video,render $USER).
    echo "kernscope dev shell (llvmPackages_21 MLIR / cudaPackages_12_6 / rocmPackages 7.2.3)"
    echo "  CC/CXX:      $CC / $CXX"
    echo "  CUDA_HOME:   $CUDA_HOME (single root)"
    echo "  CUPTI_HOME:  $CUPTI_HOME"
    echo "  mlir-tblgen: $(mlir-tblgen --version 2>&1 | head -1 || echo 'not in PATH yet')"
    echo "  nvcc:        $(nvcc --version 2>&1 | grep release || echo 'not in PATH yet')"
    echo "  hipcc:       $(hipcc --version 2>&1 | head -1 || echo 'not in PATH yet')"
    echo "  cmake:       $(cmake --version 2>&1 | head -1)"
    echo "  clang:       $(clang --version 2>&1 | head -1)"
  '';
}
