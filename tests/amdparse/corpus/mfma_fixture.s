//===- mfma_fixture.s - CDNA (gfx90a) MFMA fixture (todo-10) --------------===//
//
// Hand-authored CDNA2 (gfx90a) matrix-core fixture. Every instruction is real
// CDNA2 ISA text using the exact LLVM-MC encoding syntax (verified against
// llvm/test/MC/AMDGPU/mai-gfx90a.s, LLVM 21.1): v_mfma_f32_32x32x8f16
// (brief §5 CDNA2/3 shape), plus f16/bf16/i8 sibling shapes. The v_mfma_*
// accumulator lives in AGPRs (a[..]) per the gfx90a encoding.
//
// Used by libs/amdparse/tests/amdparse_test.cpp to assert that the parser
// classifies MFMA mnemonics with the tensor-class tag.

	.text
	.globl	test_mfma_kernel
	.p2align	2
	.type	test_mfma_kernel,@function
test_mfma_kernel:
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[2:3], -2.0
	v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[4:7]
	v_mfma_f32_32x32x4f16 a[0:31], v[0:1], v[2:3], a[34:65]
	v_mfma_i32_32x32x8i8 a[0:15], v0, v1, a[0:15]
	s_endpgm
.Lfunc_end0:
	.size	test_mfma_kernel, .Lfunc_end0-test_mfma_kernel
