// Minimal device-only CUDA source for the clang -fpass-plugin smoke
// (todo 36). Uses only clang-builtin CUDA keywords — compiled with
// -nocudainc -nocudalib (synthetic root, no CUDA toolkit needed).
extern "C" __global__ void hello_kernel() {}
