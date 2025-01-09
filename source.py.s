	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 15, 0
	.globl	_main                           ; -- Begin function main
	.p2align	2
_main:                                  ; @main
	.cfi_startproc
; %bb.0:                                ; %entry
	stp	x20, x19, [sp, #-32]!           ; 16-byte Folded Spill
	stp	x29, x30, [sp, #16]             ; 16-byte Folded Spill
	.cfi_def_cfa_offset 32
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
Lloh0:
	adrp	x0, l_.str.0@PAGE
Lloh1:
	add	x0, x0, l_.str.0@PAGEOFF
	bl	_PyString_FromString
	ldr	x8, [x0, #16]
	ldr	x8, [x8, #152]
	blr	x8
	ldr	x0, [x0, #24]
	bl	_puts
	mov	w0, #1                          ; =0x1
	bl	_PyInt_FromLong
	mov	x19, x0
	mov	w0, #1                          ; =0x1
	bl	_PyInt_FromLong
	ldr	x8, [x19, #16]
	mov	x20, x0
	mov	x0, x19
	mov	x1, x20
	ldr	x8, [x8, #48]
	blr	x8
	cbnz	x0, LBB0_2
; %bb.1:                                ; %fallback.7
	ldr	x8, [x20, #16]
	mov	x0, x20
	mov	x1, x19
	ldr	x8, [x8, #88]
	blr	x8
LBB0_2:                                 ; %end.7
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	mov	w0, wzr
	ldp	x20, x19, [sp], #32             ; 16-byte Folded Reload
	ret
	.loh AdrpAdd	Lloh0, Lloh1
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str.0:                               ; @.str.0
	.asciz	"Hello, world!"

.subsections_via_symbols
