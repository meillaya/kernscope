//===- wmma_fixture.s - RDNA3 WMMA fixture (todo-10) ----------------------===//
//
// Hand-authored RDNA3 (gfx1101) matrix-core fixture. Every instruction is
// real RDNA3 ISA text using the exact LLVM-MC encoding syntax (verified
// against llvm/test/MC/AMDGPU/gfx11_asm_wmma.s, LLVM 21.1): all six
// 16x16x16 WMMA variants per gpu-static-analyzer-research.md §5
// (v_wmma_f32_16x16x16_f16, _bf16, v_wmma_f16_16x16x16_f16,
// v_wmma_bf16_16x16x16_bf16, v_wmma_i32_16x16x16_iu8, v_wmma_i32_16x16x16_iu4).
// Wave32 register widths (v[16:23] accumulator for the half/bf16 variants,
// v[8:15] for iu8, v[4:11] for iu4) match gfx1101's default wavefrontsize32.
//
// Used by libs/amdparse/tests/amdparse_test.cpp to assert that the parser
// classifies every WMMA mnemonic with the tensor-class tag.

	.text
	.globl	test_wmma_kernel
	.p2align	2
	.type	test_wmma_kernel,@function
test_wmma_kernel:
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23]
	v_wmma_f32_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23]
	v_wmma_f16_16x16x16_f16 v[16:23], v[0:7], v[8:15], v[16:23]
	v_wmma_bf16_16x16x16_bf16 v[16:23], v[0:7], v[8:15], v[16:23]
	v_wmma_i32_16x16x16_iu8 v[8:15], v[0:3], v[4:7], v[8:15]
	v_wmma_i32_16x16x16_iu4 v[4:11], v[0:1], v[2:3], v[4:11]
	s_endpgm
.Lfunc_end0:
	.size	test_wmma_kernel, .Lfunc_end0-test_wmma_kernel
