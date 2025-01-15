	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 15, 0
	.globl	_fib                            ; -- Begin function fib
	.p2align	2
_fib:                                   ; @fib
	.cfi_startproc
; %bb.0:                                ; %entry
	stp	x20, x19, [sp, #-32]!           ; 16-byte Folded Spill
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             ; 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	.cfi_remember_state
	mov	x19, x0
	mov	x0, #1                          ; =0x1
	bl	_PyInt_FromLong
	ldr	x8, [x19, #24]
	ldr	x9, [x0, #24]
	cmp	x8, x9
	b.gt	LBB0_2
; %bb.1:                                ; %cmp.true.0
	mov	x0, #1                          ; =0x1
	bl	_PyInt_FromLong
	b	LBB0_3
LBB0_2:                                 ; %cmp.false.1
	mov	x0, xzr
	bl	_PyInt_FromLong
LBB0_3:                                 ; %cmp.end.2
	ldr	x8, [x0, #24]
	and	w8, w8, #0x1
	tbz	w8, #0, LBB0_5
; %bb.4:                                ; %if.then.3
	mov	x0, x19
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp], #32             ; 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore w19
	.cfi_restore w20
	ret
LBB0_5:                                 ; %if.else.4
	.cfi_restore_state
	b	LBB0_6
LBB0_6:                                 ; %if.end.5
	mov	x0, #1                          ; =0x1
	bl	_PyInt_FromLong
	ldr	x8, [x19, #24]
	ldr	x9, [x0, #24]
	sub	x0, x8, x9
	bl	_PyInt_FromLong
	bl	_fib
	mov	x20, x0
	mov	x0, #2                          ; =0x2
	bl	_PyInt_FromLong
	ldr	x8, [x19, #24]
	ldr	x9, [x0, #24]
	sub	x0, x8, x9
	bl	_PyInt_FromLong
	bl	_fib
	ldr	x8, [x20, #24]
	ldr	x9, [x0, #24]
	add	x0, x8, x9
	bl	_PyInt_FromLong
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	ldp	x20, x19, [sp], #32             ; 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore w19
	.cfi_restore w20
	ret
	.cfi_endproc
                                        ; -- End function
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:                                ; %entry
	stp	x29, x30, [sp, #-16]!           ; 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	w0, #30                         ; =0x1e
	bl	_PyInt_FromLong
	bl	_fib
	bl	_str
	bl	_print_object
	mov	w0, wzr
	ldp	x29, x30, [sp], #16             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
	.section	__DATA,__data
	.globl	_Py_True                        ; @Py_True
_Py_True:
	.byte	1                               ; 0x1

	.globl	_Py_False                       ; @Py_False
.zerofill __DATA,__common,_Py_False,1,0
	.globl	_Py_None                        ; @Py_None
.zerofill __DATA,__common,_Py_None,8,3
.subsections_via_symbols
