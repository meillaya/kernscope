	.amdgcn_target "amdgcn-amd-amdhsa--gfx1101"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z15layernorm_fusedPKtS0_S0_Ptfi ; -- Begin function _Z15layernorm_fusedPKtS0_S0_Ptfi
	.globl	_Z15layernorm_fusedPKtS0_S0_Ptfi
	.p2align	8
	.type	_Z15layernorm_fusedPKtS0_S0_Ptfi,@function
_Z15layernorm_fusedPKtS0_S0_Ptfi:       ; @_Z15layernorm_fusedPKtS0_S0_Ptfi
; %bb.0:
	s_load_b64 s[8:9], s[0:1], 0x20
	s_mov_b32 s3, exec_lo
	s_waitcnt lgkmcnt(0)
	v_cmpx_gt_i32_e64 s9, v0
	s_cbranch_execz .LBB0_16
; %bb.1:
	s_load_b64 s[4:5], s[0:1], 0x0
	v_mad_u64_u32 v[1:2], null, s9, s2, v[0:1]
	s_mov_b32 s2, exec_lo
	v_ashrrev_i32_e32 v2, 31, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[1:2], 1, v[1:2]
	s_waitcnt lgkmcnt(0)
	v_add_co_u32 v3, vcc_lo, s4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v4, null, s5, v2, vcc_lo
	global_load_u16 v3, v[3:4], off
	v_lshlrev_b32_e32 v4, 2, v0
	v_add_nc_u32_e32 v5, 0x200, v4
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v6, v3, v3
	ds_store_2addr_stride64_b32 v4, v3, v6 offset1:2
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_cmpx_gt_u32_e32 64, v0
	s_cbranch_execz .LBB0_3
; %bb.2:
	ds_load_2addr_stride64_b32 v[6:7], v4 offset1:1
	ds_load_2addr_stride64_b32 v[8:9], v5 offset1:1
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v6, v7, v6 :: v_dual_add_f32 v7, v9, v8
	ds_store_b32 v4, v6
	ds_store_b32 v5, v7
.LBB0_3:
	s_or_b32 exec_lo, exec_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s2, exec_lo
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_cmpx_gt_u32_e32 32, v0
	s_cbranch_execz .LBB0_5
; %bb.4:
	ds_load_2addr_b32 v[6:7], v4 offset1:32
	ds_load_2addr_b32 v[8:9], v5 offset1:32
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v6, v7, v6 :: v_dual_add_f32 v7, v9, v8
	ds_store_b32 v4, v6
	ds_store_b32 v5, v7
.LBB0_5:
	s_or_b32 exec_lo, exec_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s2, exec_lo
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_cmpx_gt_u32_e32 16, v0
	s_cbranch_execz .LBB0_7
; %bb.6:
	ds_load_2addr_b32 v[6:7], v4 offset1:16
	ds_load_2addr_b32 v[8:9], v5 offset1:16
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v6, v7, v6 :: v_dual_add_f32 v7, v9, v8
	ds_store_b32 v4, v6
	ds_store_b32 v5, v7
.LBB0_7:
	s_or_b32 exec_lo, exec_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s2, exec_lo
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_cmpx_gt_u32_e32 8, v0
	s_cbranch_execz .LBB0_9
; %bb.8:
	ds_load_2addr_b32 v[6:7], v4 offset1:8
	ds_load_2addr_b32 v[8:9], v5 offset1:8
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v6, v7, v6 :: v_dual_add_f32 v7, v9, v8
	ds_store_b32 v4, v6
	ds_store_b32 v5, v7
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s2, exec_lo
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_cmpx_gt_u32_e32 4, v0
	s_cbranch_execz .LBB0_11
; %bb.10:
	ds_load_2addr_b32 v[6:7], v4 offset1:4
	ds_load_2addr_b32 v[8:9], v5 offset1:4
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v6, v7, v6 :: v_dual_add_f32 v7, v9, v8
	ds_store_b32 v4, v6
	ds_store_b32 v5, v7
.LBB0_11:
	s_or_b32 exec_lo, exec_lo, s2
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x8
	s_load_b64 s[2:3], s[0:1], 0x18
	s_mov_b32 s0, exec_lo
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_cmpx_gt_u32_e32 2, v0
	s_cbranch_execz .LBB0_13
; %bb.12:
	ds_load_2addr_b32 v[6:7], v4 offset1:2
	ds_load_2addr_b32 v[8:9], v5 offset1:2
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v6, v7, v6 :: v_dual_add_f32 v7, v9, v8
	ds_store_b32 v4, v6
	ds_store_b32 v5, v7
.LBB0_13:
	s_or_b32 exec_lo, exec_lo, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s0, exec_lo
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_cmpx_eq_u32_e32 0, v0
	s_cbranch_execz .LBB0_15
; %bb.14:
	v_mov_b32_e32 v6, 0
	ds_load_2addr_b32 v[6:7], v6 offset0:1 offset1:129
	ds_load_b32 v8, v4
	ds_load_b32 v9, v5
	s_waitcnt lgkmcnt(0)
	v_dual_add_f32 v6, v6, v8 :: v_dual_add_f32 v7, v7, v9
	ds_store_b32 v4, v6
	ds_store_b32 v5, v7
.LBB0_15:
	s_or_b32 exec_lo, exec_lo, s0
	v_lshlrev_b32_e32 v0, 1, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	v_mov_b32_e32 v4, 0
	s_clause 0x1
	global_load_u16 v6, v0, s[4:5]
	global_load_u16 v0, v0, s[6:7]
	v_cvt_f32_u32_e32 v7, s9
	ds_load_2addr_stride64_b32 v[4:5], v4 offset1:2
	s_waitcnt lgkmcnt(0)
	v_div_scale_f32 v8, null, v7, v7, v4
	v_div_scale_f32 v9, null, v7, v7, v5
	v_div_scale_f32 v14, vcc_lo, v4, v7, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v10, v8
	v_rcp_f32_e32 v11, v9
	s_waitcnt_depctr 0xfff
	v_fma_f32 v12, -v8, v10, 1.0
	v_fma_f32 v13, -v9, v11, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v11, v13, v11 :: v_dual_fmac_f32 v10, v12, v10
	v_div_scale_f32 v12, s0, v5, v7, v5
	v_mul_f32_e32 v15, v12, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v17, -v9, v15, v12
	v_fmac_f32_e32 v15, v17, v11
	v_mul_f32_e32 v13, v14, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v9, -v9, v15, v12
	v_fma_f32 v16, -v8, v13, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v13, v16, v10
	v_fma_f32 v8, -v8, v13, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v8, v8, v10, v13
	s_mov_b32 vcc_lo, s0
	v_div_fmas_f32 v9, v9, v11, v15
	v_div_fixup_f32 v4, v8, v7, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v5, v9, v7, v5
	v_sub_f32_e32 v3, v3, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v4, v4, v5
	s_waitcnt vmcnt(0)
	v_dual_max_f32 v5, 0, v5 :: v_dual_lshlrev_b32 v0, 16, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v5, s8, v5
	v_mul_f32_e32 v7, 0x4b800000, v5
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo
	v_rsq_f32_e32 v5, v5
	s_waitcnt_depctr 0xfff
	v_mul_f32_e32 v7, 0x45800000, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e32 v4, v5, v7, vcc_lo
	v_dual_mul_f32 v3, v3, v4 :: v_dual_lshlrev_b32 v4, 16, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v0, v3, v4
	v_bfe_u32 v3, v0, 16, 1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add3_u32 v3, v0, v3, 0x7fff
	v_add_co_u32 v0, vcc_lo, s2, v1
	v_add_co_ci_u32_e64 v1, null, s3, v2, vcc_lo
	global_store_d16_hi_b16 v[0:1], v3, off
.LBB0_16:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z15layernorm_fusedPKtS0_S0_Ptfi
		.amdhsa_group_segment_fixed_size 1024
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 18
		.amdhsa_next_free_sgpr 10
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size 9
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z15layernorm_fusedPKtS0_S0_Ptfi, .Lfunc_end0-_Z15layernorm_fusedPKtS0_S0_Ptfi
                                        ; -- End function
	.set _Z15layernorm_fusedPKtS0_S0_Ptfi.num_vgpr, 18
	.set _Z15layernorm_fusedPKtS0_S0_Ptfi.num_agpr, 0
	.set _Z15layernorm_fusedPKtS0_S0_Ptfi.numbered_sgpr, 10
	.set _Z15layernorm_fusedPKtS0_S0_Ptfi.num_named_barrier, 0
	.set _Z15layernorm_fusedPKtS0_S0_Ptfi.private_seg_size, 0
	.set _Z15layernorm_fusedPKtS0_S0_Ptfi.uses_vcc, 1
	.set _Z15layernorm_fusedPKtS0_S0_Ptfi.uses_flat_scratch, 0
	.set _Z15layernorm_fusedPKtS0_S0_Ptfi.has_dyn_sized_stack, 0
	.set _Z15layernorm_fusedPKtS0_S0_Ptfi.has_recursion, 0
	.set _Z15layernorm_fusedPKtS0_S0_Ptfi.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1104
; TotalNumSgprs: 12
; NumVgprs: 18
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1024 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 12
; NumVGPRsForWavesPerEU: 18
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_dcbc95a77a9597aa,@object ; @__hip_cuid_dcbc95a77a9597aa
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_dcbc95a77a9597aa
__hip_cuid_dcbc95a77a9597aa:
	.byte	0                               ; 0x0
	.size	__hip_cuid_dcbc95a77a9597aa, 1

	.ident	"nixpkgs-AMD clang version 22.0.0 (https://github.com/ROCm/llvm-project/tree/rocm-7.2.3 rocm-7.2.3)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_dcbc95a77a9597aa
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z15layernorm_fusedPKtS0_S0_Ptfi
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         _Z15layernorm_fusedPKtS0_S0_Ptfi.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     18
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1101
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
