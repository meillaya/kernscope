// tiny_amdgcn.cpp — todo-50 AMDGCN line-table fixture source.
//
// Generated artifact (committed alongside, regenerable):
//   clang --target=amdgcn-amd-amdhsa -mcpu=gfx1101 -O3 -gline-tables-only \
//         -c tiny_amdgcn.cpp -o tiny_amdgcn.o     (source<->ISA via
//                                                  llvm-objdump --disassemble -l)
__attribute__((amdgpu_kernel)) void tiny_amdgcn_kernel(const float* in, float* out, int n) {
    int i = __builtin_amdgcn_workitem_id_x() + __builtin_amdgcn_workgroup_id_x() * 256;
    if (i < n) {
        float v = in[i] * 2.0f;
        out[i] = v + 1.0f;
    }
}
