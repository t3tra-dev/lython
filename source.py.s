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
	bl	_create_string
	bl	_print
	mov	w0, #8                          ; =0x8
	bl	_PyList_New
	mov	x19, x0
	mov	w0, wzr
	bl	_PyInt_FromI32
	mov	x1, x0
	mov	x0, x19
	bl	_PyList_Append
	mov	w0, #1                          ; =0x1
	bl	_PyInt_FromI32
	mov	x1, x0
	mov	x0, x19
	bl	_PyList_Append
	mov	w0, #2                          ; =0x2
	bl	_PyInt_FromI32
	mov	x1, x0
	mov	x0, x19
	bl	_PyList_Append
Lloh2:
	adrp	x0, l_.str.1@PAGE
Lloh3:
	add	x0, x0, l_.str.1@PAGEOFF
	bl	_create_string
	mov	x1, x0
	mov	x0, x19
	bl	_PyList_Append
Lloh4:
	adrp	x0, l_.str.2@PAGE
Lloh5:
	add	x0, x0, l_.str.2@PAGEOFF
	bl	_create_string
	mov	x1, x0
	mov	x0, x19
	bl	_PyList_Append
Lloh6:
	adrp	x0, l_.str.3@PAGE
Lloh7:
	add	x0, x0, l_.str.3@PAGEOFF
	bl	_create_string
	mov	x1, x0
	mov	x0, x19
	bl	_PyList_Append
	mov	w0, #8                          ; =0x8
	bl	_PyDict_New
	mov	x19, x0
Lloh8:
	adrp	x0, l_.str.4@PAGE
Lloh9:
	add	x0, x0, l_.str.4@PAGEOFF
	bl	_create_string
	mov	x20, x0
	mov	w0, #1                          ; =0x1
	bl	_PyInt_FromI32
	mov	x2, x0
	mov	x0, x19
	mov	x1, x20
	bl	_PyDict_SetItem
Lloh10:
	adrp	x0, l_.str.5@PAGE
Lloh11:
	add	x0, x0, l_.str.5@PAGEOFF
	bl	_create_string
	mov	x20, x0
	mov	w0, #2                          ; =0x2
	bl	_PyInt_FromI32
	mov	x2, x0
	mov	x0, x19
	mov	x1, x20
	bl	_PyDict_SetItem
Lloh12:
	adrp	x0, l_.str.6@PAGE
Lloh13:
	add	x0, x0, l_.str.6@PAGEOFF
	bl	_create_string
	mov	x20, x0
	mov	w0, #3                          ; =0x3
	bl	_PyInt_FromI32
	mov	x2, x0
	mov	x0, x19
	mov	x1, x20
	bl	_PyDict_SetItem
	ldp	x29, x30, [sp, #16]             ; 16-byte Folded Reload
	mov	w0, wzr
	ldp	x20, x19, [sp], #32             ; 16-byte Folded Reload
	ret
	.loh AdrpAdd	Lloh12, Lloh13
	.loh AdrpAdd	Lloh10, Lloh11
	.loh AdrpAdd	Lloh8, Lloh9
	.loh AdrpAdd	Lloh6, Lloh7
	.loh AdrpAdd	Lloh4, Lloh5
	.loh AdrpAdd	Lloh2, Lloh3
	.loh AdrpAdd	Lloh0, Lloh1
	.cfi_endproc
                                        ; -- End function
	.section	__TEXT,__cstring,cstring_literals
l_.str.0:                               ; @.str.0
	.asciz	"Hello, world!"

l_.str.1:                               ; @.str.1
	.asciz	"a"

l_.str.2:                               ; @.str.2
	.asciz	"b"

l_.str.3:                               ; @.str.3
	.asciz	"c"

l_.str.4:                               ; @.str.4
	.asciz	"key1"

l_.str.5:                               ; @.str.5
	.asciz	"key2"

l_.str.6:                               ; @.str.6
	.asciz	"key3"

.subsections_via_symbols
